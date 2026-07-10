from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from ultralytics import YOLO


@dataclass
class TrackCrop:
    frame_index: int
    track_id: int
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    crop_bgr: np.ndarray
    quality_score: float


class ETFFM_Lite(nn.Module):
    def __init__(self, dim_vit: int, dim_cnn: int):
        super().__init__()
        self.proj_cnn = nn.Linear(dim_cnn, dim_vit)
        self.gate = nn.Sequential(
            nn.Linear(dim_vit * 2, dim_vit),
            nn.ReLU(inplace=True),
            nn.Linear(dim_vit, dim_vit),
            nn.Sigmoid(),
        )

    def forward(self, cls_token: torch.Tensor, cnn_vec: torch.Tensor) -> torch.Tensor:
        cnn_proj = self.proj_cnn(cnn_vec)
        gate = self.gate(torch.cat([cls_token, cnn_proj], dim=1))
        return cls_token + gate * cnn_proj


class ReIDModel(nn.Module):
    """
    Compatible with the provided TEWB-style model.
    forward() returns: emb, e1, e2, lf, l1, l2
    """

    def __init__(self, num_classes: int = 245, emb_dim: int = 512):
        super().__init__()
        self.cnn = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=0)
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)

        dim_cnn = self.cnn.num_features
        dim_vit = self.vit.num_features

        self.etffm = ETFFM_Lite(dim_vit, dim_cnn)
        self.fc_final = nn.Linear(dim_vit * 2, emb_dim)
        self.cls_final = nn.Linear(emb_dim, num_classes)

        self.fc_cnn = nn.Linear(dim_cnn, emb_dim)
        self.fc_vit = nn.Linear(dim_vit, emb_dim)
        self.cls_cnn = nn.Linear(emb_dim, num_classes)
        self.cls_vit = nn.Linear(emb_dim, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cnn_fmap = self.cnn.forward_features(x)
        cnn_vec = F.adaptive_avg_pool2d(cnn_fmap, 1).flatten(1)

        x224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        vit_tokens = self.vit.forward_features(x224)
        cls_token = vit_tokens[:, 0, :]

        cls_fused = self.etffm(cls_token, cnn_vec)
        fused_feat = torch.cat([cls_token, cls_fused], dim=1)
        emb = F.normalize(self.fc_final(fused_feat), dim=1)

        e1 = F.normalize(self.fc_cnn(cnn_vec), dim=1)
        e2 = F.normalize(self.fc_vit(cls_token), dim=1)

        lf = self.cls_final(emb)
        l1 = self.cls_cnn(e1)
        l2 = self.cls_vit(e2)
        return emb, e1, e2, lf, l1, l2


class KnownGalleryManager:
    """
    Load and manage known-gallery embeddings.

    NPZ format:
      - embeddings: float32 [N, D]
      - person_ids: object[str] [N]
      - image_paths: object[str] [N]
    """

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.person_ids: List[str] = []
        self.image_paths: List[str] = []

    def load_npz(self, npz_path: str | Path) -> None:
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Gallery file not found: {path}")

        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        self.embeddings = embeddings / norms
        self.person_ids = [str(v) for v in data["person_ids"].tolist()]
        self.image_paths = [str(v) for v in data["image_paths"].tolist()]

    def save_npz(self, npz_path: str | Path) -> None:
        if self.embeddings is None:
            raise ValueError("No gallery embeddings to save.")
        np.savez_compressed(
            npz_path,
            embeddings=self.embeddings.astype(np.float32),
            person_ids=np.array(self.person_ids, dtype=object),
            image_paths=np.array(self.image_paths, dtype=object),
        )

    def build_from_directory(
        self,
        gallery_dir: str | Path,
        embed_fn,
        recursive: bool = True,
    ) -> None:
        """
        Build gallery from image folders:
        gallery_dir/
          personA/*.jpg
          personB/*.png
        """
        root = Path(gallery_dir)
        if not root.exists():
            raise FileNotFoundError(f"Gallery directory not found: {root}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = (
            [p for p in root.rglob("*") if p.suffix.lower() in exts]
            if recursive
            else [p for p in root.glob("*") if p.suffix.lower() in exts]
        )
        if not files:
            raise ValueError(f"No gallery images found in: {root}")

        embs: List[np.ndarray] = []
        pids: List[str] = []
        paths: List[str] = []

        for img_path in sorted(files):
            pid = img_path.parent.name
            emb = embed_fn(img_path)
            embs.append(emb.astype(np.float32))
            pids.append(pid)
            paths.append(str(img_path))

        stacked = np.vstack(embs).astype(np.float32)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-12
        self.embeddings = stacked / norms
        self.person_ids = pids
        self.image_paths = paths

    def search_topk(self, query_embedding: np.ndarray, topk: int = 5) -> List[Dict]:
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []

        query = query_embedding.astype(np.float32)
        query /= np.linalg.norm(query) + 1e-12
        scores = self.embeddings @ query
        topk = min(topk, len(scores))
        idx = np.argpartition(-scores, range(topk))[:topk]
        idx = idx[np.argsort(-scores[idx])]

        return [
            {
                "rank": i + 1,
                "person_id": self.person_ids[j],
                "image_path": self.image_paths[j],
                "similarity": float(scores[j]),
            }
            for i, j in enumerate(idx.tolist())
        ]


class ReIDPipelineService:
    """
    End-to-end pipeline:
    1) Track persons in video with YOLO
    2) Select robust best-quality frame crop per track
    3) Embed crop by ReID model
    4) Compare against known gallery embeddings
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolov8_person_detection.pt",
        reid_weights_path: str = "models/best_model_state_dict.pth",
        classifier_model_path: str = "models/classification.pth",
        known_threshold: float = 0.75,
        classification_known_threshold: float = 0.75,
        detection_confidence: float = 0.85,
        detection_iou: float = 0.7,
        min_box_area_ratio: float = 0.02,
        device: Optional[str] = None,
    ):
        self.project_root = Path(__file__).resolve().parents[2]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.known_threshold = known_threshold
        self.classification_known_threshold = classification_known_threshold
        self.detection_confidence = float(np.clip(detection_confidence, 0.0, 1.0))
        self.detection_iou = float(np.clip(detection_iou, 0.0, 1.0))
        self.min_box_area_ratio = max(0.0, float(min_box_area_ratio))

        self.yolo_path = self._resolve_relative_path(yolo_model_path)
        self.reid_path = self._resolve_relative_path(reid_weights_path)
        self.classifier_path = self._resolve_relative_path(classifier_model_path)

        self.yolo_model = YOLO(str(self.yolo_path))
        self.reid_model = self._load_reid_model(self.reid_path)
        self.classifier_model: Optional[nn.Module] = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.classifier_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.gallery = KnownGalleryManager()

    def _resolve_relative_path(self, relative_or_abs_path: str) -> Path:
        candidate = Path(relative_or_abs_path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def _load_reid_model(self, weights_path: Path) -> nn.Module:
        if not weights_path.exists():
            raise FileNotFoundError(f"ReID weights not found: {weights_path}")

        model = ReIDModel(num_classes=245).to(self.device)
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        drop_prefixes = ("cls_final.", "cls_cnn.", "cls_vit.")
        for key in [k for k in list(state.keys()) if k.startswith(drop_prefixes)]:
            state.pop(key, None)

        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def _load_classifier_model(self, weights_path: Path) -> nn.Module:
        if not weights_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {weights_path}")

        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=2,
        ).to(self.device)

        checkpoint = torch.load(str(weights_path), map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        if isinstance(state, nn.Module):
            state.eval()
            state.to(self.device)
            return state
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported classifier checkpoint format: {type(state)}")

        processed_state = {}
        for key, value in state.items():
            name = key[7:] if key.startswith("module.") else key
            processed_state[name] = value

        model.load_state_dict(processed_state, strict=False)
        model.eval()
        return model

    def _get_classifier_model(self) -> nn.Module:
        if self.classifier_model is None:
            self.classifier_model = self._load_classifier_model(self.classifier_path)
        return self.classifier_model

    @staticmethod
    def _expand_bbox(
        bbox_xyxy: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
        pad_x_ratio: float = 0.12,
        pad_top_ratio: float = 0.08,
        pad_bottom_ratio: float = 0.15,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox_xyxy
        frame_h, frame_w = frame_shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        pad_x = int(round(box_w * pad_x_ratio))
        pad_top = int(round(box_h * pad_top_ratio))
        pad_bottom = int(round(box_h * pad_bottom_ratio))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_top)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_bottom)
        return x1, y1, x2, y2

    @staticmethod
    def _quality_score(crop_bgr: np.ndarray, confidence: float = 1.0) -> float:
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = float(np.mean((sobel_x * sobel_x) + (sobel_y * sobel_y)))

        h, w = gray.shape[:2]
        area = float(h * w)
        min_side = float(min(h, w))
        aspect_ratio = float(w / max(h, 1))
        brightness = float(gray.mean())
        contrast = float(gray.std())

        p05, p95 = np.percentile(gray, [5.0, 95.0])
        dynamic_range = float(max(0.0, p95 - p05))

        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        hist = cv2.calcHist([gray_u8], [0], None, [64], [0, 256]).flatten().astype(np.float64)
        hist_prob = hist / (hist.sum() + 1e-12)
        entropy = float(-np.sum(hist_prob * np.log2(hist_prob + 1e-12)) / 6.0)

        median = cv2.medianBlur(gray_u8, 3).astype(np.float32)
        noise_sigma = float(np.std(gray - median))
        over_ratio = float(np.mean(gray >= 250.0))
        under_ratio = float(np.mean(gray <= 5.0))

        sharpness_score = float(np.clip(np.log1p(laplacian_var) / 7.2, 0.0, 1.0))
        texture_score = float(np.clip(np.log1p(tenengrad) / 10.2, 0.0, 1.0))
        detail_score = float(np.clip(dynamic_range / 180.0, 0.0, 1.0))
        entropy_score = float(np.clip(entropy, 0.0, 1.0))
        noise_score = float(np.clip(1.0 - (noise_sigma / 42.0), 0.0, 1.0))
        area_score = float(np.clip(np.log1p(area) / 11.5, 0.0, 1.0))
        size_score = float(np.clip(min_side / 180.0, 0.0, 1.0))
        contrast_score = float(np.clip(contrast / 70.0, 0.0, 1.0))
        brightness_score = float(np.clip(1.0 - abs(brightness - 138.0) / 138.0, 0.0, 1.0))
        clipping_score = float(np.clip(1.0 - (over_ratio + under_ratio) * 3.0, 0.0, 1.0))
        exposure_score = 0.55 * brightness_score + 0.45 * clipping_score
        aspect_score = float(np.clip(1.0 - abs(aspect_ratio - 0.42) / 0.35, 0.0, 1.0))
        confidence_score = float(np.clip(confidence, 0.0, 1.0))

        return 100.0 * (
            0.24 * sharpness_score
            + 0.19 * texture_score
            + 0.11 * detail_score
            + 0.10 * entropy_score
            + 0.09 * noise_score
            + 0.12 * area_score
            + 0.07 * size_score
            + 0.03 * contrast_score
            + 0.03 * exposure_score
            + 0.01 * aspect_score
            + 0.01 * confidence_score
        )

    @staticmethod
    def _smooth_track_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size < 3:
            return scores.copy()
        padded = np.pad(scores.astype(np.float32), (1, 1), mode="edge")
        return (padded[:-2] + (2.0 * padded[1:-1]) + padded[2:]) / 4.0

    @staticmethod
    def _bbox_area(bbox_xyxy: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _prepare_visual_crop(crop_bgr: np.ndarray, target_min_side: int = 256) -> np.ndarray:
        visual = crop_bgr.copy()
        h, w = visual.shape[:2]
        min_side = min(h, w)
        if min_side > 0 and min_side < target_min_side:
            scale = target_min_side / float(min_side)
            visual = cv2.resize(
                visual,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        blurred = cv2.GaussianBlur(visual, (0, 0), sigmaX=1.1, sigmaY=1.1)
        visual = cv2.addWeighted(visual, 1.18, blurred, -0.18, 0)
        return np.clip(visual, 0, 255).astype(np.uint8)

    def track_persons(
        self,
        video_path: str,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        min_box_area_ratio: Optional[float] = None,
    ) -> Dict[int, List[TrackCrop]]:
        """
        Track persons and collect all crops per track id.
        """
        conf = self.detection_confidence if conf is None else float(np.clip(conf, 0.0, 1.0))
        iou = self.detection_iou if iou is None else float(np.clip(iou, 0.0, 1.0))
        min_box_area_ratio = (
            self.min_box_area_ratio
            if min_box_area_ratio is None
            else max(0.0, float(min_box_area_ratio))
        )

        path = self._resolve_relative_path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        try:
            return self._track_persons_ultralytics(
                path=path,
                conf=conf,
                iou=iou,
                min_box_area_ratio=min_box_area_ratio,
            )
        except ModuleNotFoundError as exc:
            # Ultralytics tracker backend may require optional dependency "lap".
            if exc.name != "lap":
                raise
            print("[WARN] 'lap' is not installed. Falling back to built-in IoU tracker.")
            return self._track_persons_iou_fallback(
                path=path,
                conf=conf,
                iou_threshold=iou,
                min_box_area_ratio=min_box_area_ratio,
            )

    def _track_persons_ultralytics(
        self,
        path: Path,
        conf: float,
        iou: float,
        min_box_area_ratio: float,
    ) -> Dict[int, List[TrackCrop]]:
        tracks: Dict[int, List[TrackCrop]] = {}
        stream = self.yolo_model.track(
            source=str(path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=conf,
            iou=iou,
            verbose=False,
        )

        for frame_index, result in enumerate(stream):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None
            frame = result.orig_img
            frame_area = max(1, int(frame.shape[0]) * int(frame.shape[1]))

            for idx in range(len(xyxy)):
                if ids is None:
                    continue
                track_id = int(ids[idx])
                x1, y1, x2, y2 = [int(v) for v in xyxy[idx].tolist()]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if ((x2 - x1) * (y2 - y1)) / frame_area < min_box_area_ratio:
                    continue

                crop_x1, crop_y1, crop_x2, crop_y2 = self._expand_bbox((x1, y1, x2, y2), frame.shape)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                if crop.size == 0:
                    continue

                item = TrackCrop(
                    frame_index=frame_index,
                    track_id=track_id,
                    confidence=float(confs[idx]),
                    bbox_xyxy=(crop_x1, crop_y1, crop_x2, crop_y2),
                    crop_bgr=crop,
                    quality_score=self._quality_score(crop, confidence=float(confs[idx])),
                )
                tracks.setdefault(track_id, []).append(item)
        return tracks

    @staticmethod
    def _bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    def _track_persons_iou_fallback(
        self,
        path: Path,
        conf: float,
        iou_threshold: float = 0.5,
        min_box_area_ratio: float = 0.001,
        max_missed: int = 20,
    ) -> Dict[int, List[TrackCrop]]:
        """
        Tracker fallback that avoids optional 'lap' dependency.
        It uses frame-by-frame detections and greedy IoU association.
        """
        tracks: Dict[int, List[TrackCrop]] = {}
        # Active tracks: track_id -> {"bbox": (x1,y1,x2,y2), "missed": int}
        active: Dict[int, Dict[str, object]] = {}
        next_track_id = 1

        cap = cv2.VideoCapture(str(path))
        frame_index = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            result_list = self.yolo_model.predict(
                source=frame,
                conf=conf,
                iou=iou_threshold,
                classes=[0],
                verbose=False,
            )
            if not result_list:
                # No detections, increase missed count.
                for tid in list(active.keys()):
                    active[tid]["missed"] = int(active[tid]["missed"]) + 1
                    if int(active[tid]["missed"]) > max_missed:
                        del active[tid]
                frame_index += 1
                continue

            boxes = result_list[0].boxes
            detections: List[Tuple[Tuple[int, int, int, int], float]] = []
            frame_area = max(1, int(frame.shape[0]) * int(frame.shape[1]))
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                for idx in range(len(xyxy)):
                    x1, y1, x2, y2 = [int(v) for v in xyxy[idx].tolist()]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if ((x2 - x1) * (y2 - y1)) / frame_area < min_box_area_ratio:
                        continue
                    detections.append(((x1, y1, x2, y2), float(confs[idx])))

            matched_track_ids = set()
            matched_detection_ids = set()

            # Greedy match: for each active track, find best unmatched detection by IoU.
            for track_id, state in list(active.items()):
                best_det = -1
                best_iou = 0.0
                prev_box = state["bbox"]
                for det_idx, (det_box, _) in enumerate(detections):
                    if det_idx in matched_detection_ids:
                        continue
                    iou_val = self._bbox_iou(prev_box, det_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_det = det_idx

                if best_det >= 0 and best_iou >= iou_threshold:
                    det_box, det_conf = detections[best_det]
                    active[track_id]["bbox"] = det_box
                    active[track_id]["missed"] = 0
                    matched_track_ids.add(track_id)
                    matched_detection_ids.add(best_det)

                    x1, y1, x2, y2 = det_box
                    crop_x1, crop_y1, crop_x2, crop_y2 = self._expand_bbox(det_box, frame.shape)
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    if crop.size > 0:
                        item = TrackCrop(
                            frame_index=frame_index,
                            track_id=track_id,
                            confidence=det_conf,
                            bbox_xyxy=(crop_x1, crop_y1, crop_x2, crop_y2),
                            crop_bgr=crop,
                            quality_score=self._quality_score(crop, confidence=det_conf),
                        )
                        tracks.setdefault(track_id, []).append(item)
                else:
                    active[track_id]["missed"] = int(active[track_id]["missed"]) + 1
                    if int(active[track_id]["missed"]) > max_missed:
                        del active[track_id]

            # Create new tracks for unmatched detections.
            for det_idx, (det_box, det_conf) in enumerate(detections):
                if det_idx in matched_detection_ids:
                    continue
                track_id = next_track_id
                next_track_id += 1
                active[track_id] = {"bbox": det_box, "missed": 0}

                x1, y1, x2, y2 = det_box
                crop_x1, crop_y1, crop_x2, crop_y2 = self._expand_bbox(det_box, frame.shape)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                if crop.size > 0:
                    item = TrackCrop(
                        frame_index=frame_index,
                        track_id=track_id,
                        confidence=det_conf,
                        bbox_xyxy=(crop_x1, crop_y1, crop_x2, crop_y2),
                        crop_bgr=crop,
                        quality_score=self._quality_score(crop, confidence=det_conf),
                    )
                    tracks.setdefault(track_id, []).append(item)

            frame_index += 1

        cap.release()
        return tracks

    def select_best_frames(self, tracks: Dict[int, List[TrackCrop]]) -> Dict[int, TrackCrop]:
        """
        Select the most reliable high-quality crop for each track.
        We smooth per-frame quality and rank candidates with a robust score
        to avoid one-frame outliers.
        """
        best: Dict[int, TrackCrop] = {}
        for track_id, items in tracks.items():
            if not items:
                continue
            if len(items) == 1:
                best[track_id] = items[0]
                continue

            ordered = sorted(items, key=lambda x: x.frame_index)
            raw_quality = np.array([max(0.0, float(it.quality_score)) for it in ordered], dtype=np.float32)
            smooth_quality = self._smooth_track_scores(raw_quality)

            quality_floor = float(np.percentile(smooth_quality, 30.0))
            candidate_indices = [i for i, q in enumerate(smooth_quality.tolist()) if q >= quality_floor]
            if not candidate_indices:
                candidate_indices = list(range(len(ordered)))

            candidate_quality = smooth_quality[candidate_indices]
            q_min = float(candidate_quality.min())
            q_max = float(candidate_quality.max())
            q_span = q_max - q_min

            frame_idx = np.array([float(it.frame_index) for it in ordered], dtype=np.float32)
            frame_min = float(frame_idx.min())
            frame_max = float(frame_idx.max())
            frame_center = 0.5 * (frame_min + frame_max)
            frame_half_span = max(1.0, 0.5 * (frame_max - frame_min))

            areas = np.array([float(self._bbox_area(it.bbox_xyxy)) for it in ordered], dtype=np.float32)
            area_ref = max(float(np.percentile(areas, 90.0)), 1.0)

            chosen_idx = candidate_indices[0]
            chosen_score = -1.0
            for idx in candidate_indices:
                if q_span < 1e-6:
                    quality_norm = 1.0
                else:
                    quality_norm = float(np.clip((smooth_quality[idx] - q_min) / q_span, 0.0, 1.0))

                temporal_score = float(
                    np.clip(1.0 - abs(float(frame_idx[idx]) - frame_center) / frame_half_span, 0.0, 1.0)
                )
                area_score = float(np.clip(np.log1p(float(areas[idx])) / np.log1p(area_ref), 0.0, 1.0))
                conf_score = float(np.clip(float(ordered[idx].confidence), 0.0, 1.0))

                robust_score = (
                    0.68 * quality_norm
                    + 0.14 * temporal_score
                    + 0.10 * area_score
                    + 0.08 * conf_score
                )

                if robust_score > chosen_score:
                    chosen_score = robust_score
                    chosen_idx = idx

            best[track_id] = ordered[chosen_idx]
        return best

    def _save_track_crop(self, crop: TrackCrop, output_dir: Path) -> str:
        """
        Save cropped image for a selected track frame and return its path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"track_{crop.track_id:04d}_frame_{crop.frame_index:06d}.jpg"
        output_path = output_dir / file_name
        visual_crop = self._prepare_visual_crop(crop.crop_bgr)
        ok = cv2.imwrite(str(output_path), visual_crop)
        if not ok:
            raise RuntimeError(f"Failed to save track crop: {output_path}")
        return output_path.as_posix()

    def show_best_per_track_images(
        self,
        best_per_track: Dict[int, TrackCrop],
        max_items: int = 30,
        cols: int = 5,
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ):
        """
        Visualize selected best crop for each track.
        """
        import math
        import matplotlib.pyplot as plt

        if not best_per_track:
            print("[INFO] No best-per-track images to show.")
            return None

        items = sorted(best_per_track.items(), key=lambda x: x[0])[:max_items]
        n = len(items)
        cols = max(1, cols)
        rows = int(math.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        flat_axes = axes.flatten()
        for idx, (track_id, crop) in enumerate(items):
            ax = flat_axes[idx]
            rgb = cv2.cvtColor(self._prepare_visual_crop(crop.crop_bgr), cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(
                f"track={track_id}\nframe={crop.frame_index} q={crop.quality_score:.2f}",
                fontsize=9,
            )
            ax.axis("off")

        for idx in range(n, len(flat_axes)):
            flat_axes[idx].axis("off")

        plt.tight_layout()
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=180, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    @torch.no_grad()
    def embed_image(self, image_path: str | Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb, _, _, _, _, _ = self.reid_model(tensor)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb, _, _, _, _, _ = self.reid_model(tensor)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def classify_crop_known_unknown(self, crop_bgr: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.classifier_transform(img).unsqueeze(0).to(self.device)
        logits = self._get_classifier_model()(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        probs = torch.softmax(logits, dim=1).squeeze(0)
        known_prob = float(probs[0].item())
        unknown_prob = float(probs[1].item())
        is_known = known_prob >= self.classification_known_threshold
        confidence = known_prob if is_known else unknown_prob

        return {
            "label": "known" if is_known else "unknown",
            "display_label": "Quen" if is_known else "La",
            "classification_confidence": confidence,
            "threshold": self.classification_known_threshold,
            "classification_probs": {
                "known": known_prob,
                "unknown": unknown_prob,
            },
        }

    def load_known_gallery(self, gallery_npz_path: str = "known_gallery/gallery_embeddings.npz") -> None:
        path = self._resolve_relative_path(gallery_npz_path)
        self.gallery.load_npz(path)

    def build_known_gallery(
        self,
        gallery_images_dir: str = "known_gallery/images",
        output_npz_path: str = "known_gallery/gallery_embeddings.npz",
    ) -> None:
        images_dir = self._resolve_relative_path(gallery_images_dir)
        output_path = self._resolve_relative_path(output_npz_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.gallery.build_from_directory(images_dir, embed_fn=self.embed_image, recursive=True)
        self.gallery.save_npz(output_path)

    def compare_with_gallery(self, query_embedding: np.ndarray, topk: int = 5) -> Dict:
        top_matches = self.gallery.search_topk(query_embedding=query_embedding, topk=topk)
        if not top_matches:
            return {
                "label": "unknown",
                "person_id": None,
                "best_similarity": 0.0,
                "topk": [],
            }

        best = top_matches[0]
        is_known = best["similarity"] >= self.known_threshold
        return {
            "label": "known" if is_known else "unknown",
            "person_id": best["person_id"] if is_known else "unknown",
            "best_similarity": float(best["similarity"]),
            "threshold": self.known_threshold,
            "topk": top_matches,
        }

    def format_match_result(self, raw_result: Dict, decimals: int = 4) -> Dict:
        """
        Format result from compare_with_gallery/process_query_image into a readable schema.
        """
        topk_items = raw_result.get("topk", []) or []
        return {
            "label": raw_result.get("label", "unknown"),
            "person_id": raw_result.get("person_id", "unknown"),
            "best_similarity": round(float(raw_result.get("best_similarity", 0.0)), decimals),
            "threshold": float(raw_result.get("threshold", self.known_threshold)),
            "topk": [
                {
                    "rank": int(item.get("rank", idx + 1)),
                    "person_id": str(item.get("person_id", "unknown")),
                    "similarity": round(float(item.get("similarity", 0.0)), decimals),
                    "image_path": str(item.get("image_path", "")),
                }
                for idx, item in enumerate(topk_items)
            ],
        }

    def format_video_result(self, video_result: Dict, decimals: int = 4) -> Dict:
        """
        Format result from process_video() with rounded scores and concise keys.
        """
        formatted_items: List[Dict] = []
        for item in video_result.get("results", []):
            prediction = self.format_match_result(item.get("prediction", {}), decimals=decimals)
            raw_pred = item.get("prediction", {}) or {}
            if raw_pred.get("mode"):
                prediction["mode"] = str(raw_pred.get("mode"))
            if raw_pred.get("classification_confidence") is not None:
                prediction["classification_confidence"] = round(
                    float(raw_pred.get("classification_confidence")), decimals
                )
            if raw_pred.get("display_label"):
                prediction["display_label"] = str(raw_pred.get("display_label"))
            if raw_pred.get("classification_probs"):
                probs = raw_pred.get("classification_probs") or {}
                prediction["classification_probs"] = {
                    "known": round(float(probs.get("known", 0.0)), decimals),
                    "unknown": round(float(probs.get("unknown", 0.0)), decimals),
                }
            formatted_items.append(
                {
                    "track_id": int(item.get("track_id", -1)),
                    "frame_index": int(item.get("frame_index", -1)),
                    "bbox_xyxy": item.get("bbox_xyxy"),
                    "track_url": str(item.get("track_url", "")),
                    "quality_score": round(float(item.get("quality_score", 0.0)), decimals),
                    "prediction": prediction,
                }
            )

        return {
            "video_path": str(video_result.get("video_path", "")),
            "num_tracks": int(video_result.get("num_tracks", len(formatted_items))),
            "threshold": float(video_result.get("threshold", self.known_threshold)),
            "mode": str(video_result.get("mode", "similarity")),
            "results": formatted_items,
        }

    def show_topk_with_similarity(
        self,
        query_image_path: str | Path,
        match_result: Dict,
        topk: int = 5,
        save_path: Optional[str | Path] = None,
        show: bool = True,
    ):
        """
        Show query image and top-k gallery images with similarity scores.
        """
        import matplotlib.pyplot as plt

        query_path = Path(query_image_path)
        if not query_path.exists():
            raise FileNotFoundError(f"Query image not found: {query_path}")

        topk_items = (match_result.get("topk", []) or [])[:topk]
        cols = max(1, len(topk_items))

        fig = plt.figure(figsize=(3.5 * cols, 6))
        grid = fig.add_gridspec(2, cols, height_ratios=[2.2, 1.8])

        ax_query = fig.add_subplot(grid[0, :])
        q_img = Image.open(query_path).convert("RGB")
        ax_query.imshow(q_img)
        ax_query.set_title(
            f"Query: {query_path.name} | label={match_result.get('label')} "
            f"| best={float(match_result.get('best_similarity', 0.0)):.4f}",
            fontsize=12,
        )
        ax_query.axis("off")

        for idx in range(cols):
            ax = fig.add_subplot(grid[1, idx])
            if idx < len(topk_items):
                item = topk_items[idx]
                candidate_path = Path(str(item.get("image_path", "")))
                if candidate_path.exists():
                    g_img = Image.open(candidate_path).convert("RGB")
                    ax.imshow(g_img)
                else:
                    ax.imshow(np.full((128, 64, 3), 240, dtype=np.uint8))

                ax.set_title(
                    f"Top {idx + 1}\n"
                    f"ID={item.get('person_id', 'unknown')}\n"
                    f"sim={float(item.get('similarity', 0.0)):.4f}",
                    fontsize=10,
                )
            else:
                ax.imshow(np.full((128, 64, 3), 240, dtype=np.uint8))
                ax.set_title(f"Top {idx + 1}\n(empty)", fontsize=10)
            ax.axis("off")

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=180, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def process_query_image(self, query_image_path: str, topk: int = 5) -> Dict:
        """
        Query-only path: one image in, top-k ranking out.
        """
        query_path = self._resolve_relative_path(query_image_path)
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

        if not query_path.exists():
            raise FileNotFoundError(f"Query image not found: {query_path}")
        suffix = query_path.suffix.lower()
        if suffix in video_exts:
            raise ValueError(
                f"Received video file '{query_path.name}' in process_query_image(). "
                "Use process_video(...) for video input."
            )
        if suffix and suffix not in image_exts:
            raise ValueError(
                f"Unsupported query image extension '{suffix}'. "
                f"Supported image extensions: {sorted(image_exts)}"
            )

        try:
            emb = self.embed_image(query_path)
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"Cannot decode image file: {query_path}. "
                "Verify the file is a valid image format."
            ) from exc

        result = self.compare_with_gallery(emb, topk=topk)
        result["query_image"] = str(query_path)
        return result

    def process_video(
        self,
        video_path: str,
        topk: int = 5,
        use_classification_model: bool = False,
        detect_confidence: Optional[float] = None,
        detect_iou: Optional[float] = None,
        min_box_area_ratio: Optional[float] = None,
        show_best_frames: bool = False,
        best_frames_save_path: Optional[str | Path] = None,
        best_frames_max_items: int = 30,
        track_frames_dir: str | Path = "result_frame",
    ) -> Dict:
        """
        Full video pipeline.
        """
        tracks = self.track_persons(
            video_path=video_path,
            conf=detect_confidence,
            iou=detect_iou,
            min_box_area_ratio=min_box_area_ratio,
        )
        best_per_track = self.select_best_frames(tracks)
        track_frames_path = self._resolve_relative_path(str(track_frames_dir))
        track_frames_path.mkdir(parents=True, exist_ok=True)
        if show_best_frames:
            self.show_best_per_track_images(
                best_per_track=best_per_track,
                max_items=best_frames_max_items,
                save_path=best_frames_save_path,
                show=True,
            )

        predictions = []
        for track_id, crop in sorted(best_per_track.items()):
            if use_classification_model:
                cls_result = self.classify_crop_known_unknown(crop.crop_bgr)
                ranked = {
                    "mode": "classification",
                    "label": cls_result.get("label", "unknown"),
                    "person_id": "known" if cls_result.get("label") == "known" else "unknown",
                    "best_similarity": float(cls_result.get("classification_confidence", 0.0)),
                    "classification_confidence": float(cls_result.get("classification_confidence", 0.0)),
                    "classification_probs": cls_result.get("classification_probs", {}),
                    "display_label": cls_result.get("display_label", "La"),
                    "threshold": float(
                        cls_result.get("threshold", self.classification_known_threshold)
                    ),
                    "topk": [],
                }
            else:
                emb = self.embed_crop(crop.crop_bgr)
                ranked = self.compare_with_gallery(emb, topk=topk)
                ranked["mode"] = "similarity"
            track_url = self._save_track_crop(crop, track_frames_path)
            predictions.append(
                {
                    "track_id": track_id,
                    "frame_index": crop.frame_index,
                    "bbox_xyxy": crop.bbox_xyxy,
                    "track_url": track_url,
                    "quality_score": float(crop.quality_score),
                    "prediction": ranked,
                }
            )

        return {
            "video_path": str(self._resolve_relative_path(video_path)),
            "num_tracks": len(best_per_track),
            "threshold": self.classification_known_threshold if use_classification_model else self.known_threshold,
            "mode": "classification" if use_classification_model else "similarity",
            "results": predictions,
        }


if __name__ == "__main__":
    """
    Demo script for ReIDPipelineService

    Usage:
    1) Build known gallery from images in known_persons/
    2) Process a video and match against known persons
    """
    import sys

    print("=" * 60)
    print("ReID Pipeline Service - Demo")
    print("=" * 60)

    # Initialize service
    print("\n1️⃣  Initializing ReID service...")
    service = ReIDPipelineService(
        yolo_model_path="models/yolov8_person_detection.pt",
        reid_weights_path="models/best_model_state_dict.pth",
        known_threshold=0.75
    )
    print("✅ Service initialized")

    # Check if known_gallery folder exists
    import os
    if not os.path.exists("known_gallery"):
        print("\n⚠️  'known_gallery' folder not found")
        print("Creating example structure...")
        os.makedirs("known_gallery", exist_ok=True)
        print("Please add person images to 'known_gallery/' folder")
        print("Format: known_gallery/<person_name>/<images>.jpg")
        sys.exit(0)

    # Build or load known gallery
    gallery_path = "known_gallery/gallery_embeddings.npz"

    if not os.path.exists(gallery_path):
        print("\n2️⃣  Building known gallery from images...")
        try:
            service.build_known_gallery("known_gallery", gallery_path)
            print(f"✅ Gallery saved to: {gallery_path}")
        except Exception as e:
            print(f"❌ Error building gallery: {e}")
            sys.exit(1)
    else:
        print(f"\n2️⃣  Loading existing gallery: {gallery_path}")
        service.load_known_gallery(gallery_path)
        print("✅ Gallery loaded")

    # Example: Process a query image
    print("\n3️⃣  Demo options:")
    print("   - To process an image: service.process_query_image('image.jpg', topk=5)")
    print("   - To process a video: service.process_video('video.mp4', topk=5)")
    print("\n📝 Edit the code in 'if __name__ == \"__main__\"' block to customize")

    # Example video processing (commented out - add your video path)
    # print("\n4️⃣  Processing video...")
    # video_path = "temp/test_video.mp4"  # Change this
    # if os.path.exists(video_path):
    #     raw = service.process_video(video_path, topk=5)
    #     formatted = service.format_video_result(raw)
    #     print("\n📊 Results:")
    #     print(formatted)
    # else:
    #     print(f"⚠️  Video not found: {video_path}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
