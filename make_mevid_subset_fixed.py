#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_mevid_subset_fixed.py

Tạo subset MEVID test cho open-set experiments theo hướng "luận văn":
- Chọn KNOWN_PIDS (known identities)
- Với mỗi known PID:
    - chọn KNOWN_QUERY_PER_PID tracklets làm query (positive)
    - chọn tối đa MAX_GALLERY_TRACKLETS tracklets làm gallery
- Với unknown PIDs:
    - chọn tối đa MAX_UNKNOWN_QUERY_TRACKLETS tracklets làm query (negative)
- Mỗi tracklet: sample tối đa MAX_FRAMES_PER_TRACKLET frames (evenly)
- Copy ảnh sang dst/bbox_test/<pid4>/<filename>
- Rebuild:
    - test_name.txt
    - track_test_info.txt (dòng i ứng với tracklet i, start/end contiguous)
    - query_IDX.txt (indices của query tracklets trong track_test_info mới)
    - gallery_IDX.txt (indices của gallery tracklets trong track_test_info mới)
    - split_known.txt / split_unknown.txt (PID lists)

Chế độ xử lý ảnh thiếu:
- STRICT_MODE=True: thiếu 1 ảnh -> raise error
- STRICT_MODE=False: bỏ frame thiếu, bỏ tracklet nếu:
    - còn < MIN_FRAMES_PER_TRACKLET
    - hoặc missing_frac > MAX_MISSING_FRAC

Ghi log ảnh thiếu: missing_images.txt
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


# -------------------------
# Utils
# -------------------------

def parse_sci_int(x: str) -> int:
    x = x.strip()
    if not x:
        raise ValueError("Empty token while parsing numeric value.")
    return int(round(float(x)))


def read_lines(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


def sample_evenly(items: List[str], k: int) -> List[str]:
    """Sample up to k elements evenly across list."""
    n = len(items)
    if k <= 0:
        return []
    if n <= k:
        return items[:]
    if k == 1:
        return [items[n // 2]]
    out = []
    for i in range(k):
        pos = int(round(i * (n - 1) / (k - 1)))
        out.append(items[pos])

    # Dedup due to rounding
    dedup, seen = [], set()
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def infer_pid4_from_filename(fname: str) -> str:
    # MEVID naming rule: first 4 digits are PID with leading zeros (e.g., 0201)
    if len(fname) >= 4 and fname[:4].isdigit():
        return fname[:4]
    return "unknown_pid"


# -------------------------
# Load MEVID annotation
# -------------------------

def load_test_name(src_dir: Path) -> List[str]:
    p = src_dir / "test_name.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    names = read_lines(p)
    if not names:
        raise ValueError("test_name.txt is empty.")
    return names


def load_track_test_info(src_dir: Path) -> List[Tuple[int, int, int, int, int]]:
    p = src_dir / "track_test_info.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    rows: List[Tuple[int, int, int, int, int]] = []
    for line in read_lines(p):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid track_test_info line (need 5 cols): {line}")
        start = parse_sci_int(parts[0])
        end = parse_sci_int(parts[1])
        pid = parse_sci_int(parts[2])
        outfit = parse_sci_int(parts[3])
        cam = parse_sci_int(parts[4])
        if end < start:
            raise ValueError(f"Invalid range start>end in line: {line}")
        rows.append((start, end, pid, outfit, cam))
    if not rows:
        raise ValueError("track_test_info.txt is empty.")
    return rows


def build_tracklets(test_names: List[str], track_rows: List[Tuple[int, int, int, int, int]]) -> List[Dict[str, Any]]:
    """Tracklet i corresponds to line i in track_test_info.txt. images slice test_name[start:end+1]."""
    n_names = len(test_names)
    tracklets: List[Dict[str, Any]] = []
    for tid, (start, end, pid, outfit, cam) in enumerate(track_rows):
        if start < 0 or end >= n_names:
            raise ValueError(f"Tracklet {tid} out of range: start={start}, end={end}, n_names={n_names}")
        imgs = test_names[start:end + 1]
        tracklets.append({
            "tracklet_id": tid,
            "start": start,
            "end": end,
            "person_id": pid,
            "outfit_id": outfit,
            "camera_id": cam,
            "images": imgs,  # entries as in test_name.txt (often filename only)
        })
    return tracklets


def pid_to_tracklets(tracklets: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    m: Dict[int, List[int]] = {}
    for t in tracklets:
        m.setdefault(t["person_id"], []).append(t["tracklet_id"])
    return m


# -------------------------
# File locating + copying
# -------------------------

def build_filename_index(src_bbox_dir: Path) -> Dict[str, Path]:
    """Build filename -> Path index (fallback). With MEVID naming, filenames are typically unique."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    idx: Dict[str, Path] = {}
    for p in src_bbox_dir.rglob("*"):
        if p.is_file() and p.suffix in exts:
            idx[p.name] = p
    return idx


def locate_src_image(src_bbox_dir: Path, name_in_test: str, filename_index: Optional[Dict[str, Path]] = None) -> Optional[Path]:
    """
    Locate image given entry in test_name.txt (often filename only).
    Prefer bbox_test/<pid4>/<filename>. Fallback to global index by filename.
    """
    fname = Path(name_in_test).name
    pid4 = infer_pid4_from_filename(fname)
    p = src_bbox_dir / pid4 / fname
    if p.exists():
        return p
    if filename_index is not None:
        return filename_index.get(fname)
    return None


def copy_one(src_path: Path, dst_bbox_dir: Path) -> Path:
    fname = src_path.name
    pid4 = infer_pid4_from_filename(fname)
    dst_path = dst_bbox_dir / pid4 / fname
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)
    return dst_path


# -------------------------
# Main
# -------------------------

def main():
    # =========================
    # HARD-CODED CONFIG (tự chỉnh ở đây)
    # =========================
    SRC_MEVID_DIR = Path(r"V:\Doc\THUCTAP\DTH_Indentify\mevid-v1-annotation-data")
    DST_MEVID_DIR = Path(r"V:\Doc\THUCTAP\DTH_Indentify\mevid_subset_fixed")

    # ---- Open-set split config (luận văn) ----
    KNOWN_PIDS = 20

    KNOWN_QUERY_PER_PID = 1        # 1 tracklet query / known PID (positive)
    MAX_GALLERY_TRACKLETS = 8      # max gallery tracklets / known PID

    MAX_UNKNOWN_QUERY_TRACKLETS = 100  # max unknown query tracklets total (negative)

    # ---- Sampling ----
    MAX_TRACKLETS_PER_PID = 10     # cap tracklets considered per PID (for both known & unknown)
    MAX_FRAMES_PER_TRACKLET = 60   # sample evenly

    PREFER_LONG_TRACKLETS = True
    RANDOM_SEED = 42

    # ---- Missing data handling ----
    STRICT_MODE = False            # True: thiếu 1 frame -> dừng; False: drop frames
    MIN_FRAMES_PER_TRACKLET = 20   # tracklet còn < 20 frame -> bỏ
    MAX_MISSING_FRAC = 0.7         # mất > 50% frame -> bỏ tracklet

    MISSING_LOG_NAME = "missing_images.txt"
    # =========================
    # END CONFIG
    # =========================

    random.seed(RANDOM_SEED)

    src_mevid = SRC_MEVID_DIR
    dst_mevid = DST_MEVID_DIR

    src_bbox = src_mevid / "bbox_test"
    if not src_bbox.exists():
        raise FileNotFoundError(f"Missing bbox_test directory: {src_bbox}")

    # Load annotations
    test_names = load_test_name(src_mevid)
    track_rows = load_track_test_info(src_mevid)
    tracklets = build_tracklets(test_names, track_rows)

    pid_map = pid_to_tracklets(tracklets)
    all_pids = sorted(pid_map.keys())

    # Chọn known pids từ các PID có đủ tracklet để tách query/gallery (>=2 tracklets)
    eligible_known = [pid for pid in all_pids if len(pid_map.get(pid, [])) >= (KNOWN_QUERY_PER_PID + 1)]
    if len(eligible_known) < KNOWN_PIDS:
        # fallback: vẫn chọn từ all_pids nhưng sẽ có PID không đủ query/gallery
        eligible_known = all_pids[:]

    if len(eligible_known) < KNOWN_PIDS:
        raise ValueError(f"Not enough pids in test set: total={len(all_pids)}, known_pids={KNOWN_PIDS}")

    known_pids = set(random.sample(eligible_known, KNOWN_PIDS))
    unknown_pids = [pid for pid in all_pids if pid not in known_pids]

    # Prepare dst structure
    if dst_mevid.exists():
        raise FileExistsError(f"dst_mevid_dir already exists. Please delete it or choose another: {dst_mevid}")
    dst_mevid.mkdir(parents=True, exist_ok=True)
    dst_bbox = dst_mevid / "bbox_test"
    dst_bbox.mkdir(parents=True, exist_ok=True)

    missing_log_path = dst_mevid / MISSING_LOG_NAME

    filename_index = build_filename_index(src_bbox)

    print(f"[INFO] Source: {src_mevid}")
    print(f"[INFO] n_images(test_name): {len(test_names)}")
    print(f"[INFO] n_tracklets(track_test_info): {len(track_rows)}")
    print(f"[INFO] Total PIDs in source test: {len(all_pids)}")
    print(f"[INFO] KNOWN PIDs: {len(known_pids)} | UNKNOWN PIDs: {len(unknown_pids)}")
    print(f"[INFO] Mode: {'STRICT' if STRICT_MODE else 'DROP_MISSING+QC'}")
    print(f"[INFO] QC: MIN_FRAMES_PER_TRACKLET={MIN_FRAMES_PER_TRACKLET}, MAX_MISSING_FRAC={MAX_MISSING_FRAC}")
    print("--------------------------------------------------")

    # --------- Build selection lists (by role) ----------
    # Each element will become a tracklet row in new track_test_info
    selected_gallery: List[Dict[str, Any]] = []
    selected_query: List[Dict[str, Any]] = []

    # Helper: order tracklets ids for a pid
    def order_tracklets_for_pid(pid: int) -> List[int]:
        tids = pid_map[pid][:]
        if PREFER_LONG_TRACKLETS:
            tids.sort(key=lambda tid: len(tracklets[tid]["images"]), reverse=True)
        else:
            random.shuffle(tids)
        return tids[:MAX_TRACKLETS_PER_PID]

    # Known PIDs: pick query then gallery
    for pid in sorted(known_pids):
        tids = order_tracklets_for_pid(pid)
        if len(tids) < (KNOWN_QUERY_PER_PID + 1):
            # Không đủ để vừa query vừa gallery -> chỉ lấy gallery nếu có
            for tid in tids[:MAX_GALLERY_TRACKLETS]:
                t = tracklets[tid]
                sampled = sample_evenly(t["images"], MAX_FRAMES_PER_TRACKLET)
                if sampled:
                    selected_gallery.append({
                        "role": "gallery",
                        "person_id": t["person_id"],
                        "outfit_id": t["outfit_id"],
                        "camera_id": t["camera_id"],
                        "images": sampled,
                    })
            continue

        # Query tracklets (positive)
        for tid in tids[:KNOWN_QUERY_PER_PID]:
            t = tracklets[tid]
            sampled = sample_evenly(t["images"], MAX_FRAMES_PER_TRACKLET)
            if sampled:
                selected_query.append({
                    "role": "query_known",
                    "person_id": t["person_id"],
                    "outfit_id": t["outfit_id"],
                    "camera_id": t["camera_id"],
                    "images": sampled,
                })

        # Gallery tracklets
        for tid in tids[KNOWN_QUERY_PER_PID:KNOWN_QUERY_PER_PID + MAX_GALLERY_TRACKLETS]:
            t = tracklets[tid]
            sampled = sample_evenly(t["images"], MAX_FRAMES_PER_TRACKLET)
            if sampled:
                selected_gallery.append({
                    "role": "gallery",
                    "person_id": t["person_id"],
                    "outfit_id": t["outfit_id"],
                    "camera_id": t["camera_id"],
                    "images": sampled,
                })

    # Unknown PIDs: pick query tracklets up to MAX_UNKNOWN_QUERY_TRACKLETS total
    unknown_added = 0
    for pid in unknown_pids:
        if unknown_added >= MAX_UNKNOWN_QUERY_TRACKLETS:
            break
        tids = order_tracklets_for_pid(pid)
        for tid in tids:
            if unknown_added >= MAX_UNKNOWN_QUERY_TRACKLETS:
                break
            t = tracklets[tid]
            sampled = sample_evenly(t["images"], MAX_FRAMES_PER_TRACKLET)
            if sampled:
                selected_query.append({
                    "role": "query_unknown",
                    "person_id": t["person_id"],
                    "outfit_id": t["outfit_id"],
                    "camera_id": t["camera_id"],
                    "images": sampled,
                })
                unknown_added += 1

    # Merge all selected tracklets into a single list (order: gallery first, then query)
    selected_all = selected_gallery + selected_query

    print(f"[INFO] Selected gallery tracklets: {len(selected_gallery)}")
    print(f"[INFO] Selected query tracklets: {len(selected_query)}  (unknown capped at {MAX_UNKNOWN_QUERY_TRACKLETS})")
    print(f"[INFO] Total selected tracklets (before copy/QC): {len(selected_all)}")
    print("--------------------------------------------------")

    # --------- Rebuild outputs contiguously while copying ----------
    rebuilt_test_name: List[str] = []
    rebuilt_track_rows: List[Tuple[int, int, int, int, int]] = []
    query_idx: List[int] = []
    gallery_idx: List[int] = []

    missing: List[str] = []

    # Stats
    copied_unique = 0
    seen_dst = set()
    dropped_tracklets_too_short = 0
    dropped_tracklets_too_missing = 0
    total_missing_frames = 0

    for sel in selected_all:
        kept_imgs: List[str] = []
        missing_in_tracklet = 0
        total_in_tracklet = len(sel["images"])

        for rel in sel["images"]:
            src_path = locate_src_image(src_bbox, rel, filename_index=filename_index)

            if src_path is None or not src_path.exists():
                miss_name = Path(rel).name
                missing.append(miss_name)
                missing_in_tracklet += 1
                total_missing_frames += 1

                if STRICT_MODE:
                    missing_log_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
                    raise FileNotFoundError(f"[STRICT] Missing image: {miss_name} (logged to {missing_log_path})")

                # DROP_MISSING: skip this frame
                continue

            dst_path = copy_one(src_path, dst_bbox)

            if str(dst_path) not in seen_dst:
                seen_dst.add(str(dst_path))
                copied_unique += 1

            kept_imgs.append(dst_path.name)  # store filename only

        # QC decisions (only relevant when not strict)
        missing_frac = (missing_in_tracklet / total_in_tracklet) if total_in_tracklet > 0 else 1.0

        if not STRICT_MODE:
            if missing_frac > MAX_MISSING_FRAC:
                dropped_tracklets_too_missing += 1
                continue
            if len(kept_imgs) < MIN_FRAMES_PER_TRACKLET:
                dropped_tracklets_too_short += 1
                continue
        else:
            # strict would have errored earlier if any missing; still guard
            if not kept_imgs:
                continue

        # Keep tracklet
        start = len(rebuilt_test_name)
        rebuilt_test_name.extend(kept_imgs)
        end = len(rebuilt_test_name) - 1

        new_tid = len(rebuilt_track_rows)
        rebuilt_track_rows.append((start, end, sel["person_id"], sel["outfit_id"], sel["camera_id"]))

        if sel["role"] == "gallery":
            gallery_idx.append(new_tid)
        else:
            query_idx.append(new_tid)

    # Write missing log
    missing_log_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    # Write outputs
    (dst_mevid / "test_name.txt").write_text("\n".join(rebuilt_test_name) + "\n", encoding="utf-8")
    (dst_mevid / "track_test_info.txt").write_text(
        "\n".join(f"{s} {e} {pid} {oid} {cid}" for (s, e, pid, oid, cid) in rebuilt_track_rows) + "\n",
        encoding="utf-8",
    )
    (dst_mevid / "query_IDX.txt").write_text("\n".join(str(i) for i in query_idx) + "\n", encoding="utf-8")
    (dst_mevid / "gallery_IDX.txt").write_text("\n".join(str(i) for i in gallery_idx) + "\n", encoding="utf-8")

    (dst_mevid / "split_known.txt").write_text("\n".join(str(pid) for pid in sorted(known_pids)) + "\n", encoding="utf-8")
    (dst_mevid / "split_unknown.txt").write_text("\n".join(str(pid) for pid in unknown_pids) + "\n", encoding="utf-8")

    print("--------------------------------------------------")
    print(f"[DONE] Wrote subset to: {dst_mevid}")
    print(f"  - bbox_test/: {dst_bbox}")
    print(f"  - test_name.txt lines: {len(rebuilt_test_name)}")
    print(f"  - track_test_info.txt tracklets: {len(rebuilt_track_rows)}")
    print(f"  - gallery_IDX.txt: {len(gallery_idx)}")
    print(f"  - query_IDX.txt: {len(query_idx)}")
    print(f"  - copied unique images: {copied_unique}")
    print(f"  - missing images: {len(missing)}  (log: {missing_log_path})")

    if not STRICT_MODE:
        print(f"[STATS] Missing frames total: {total_missing_frames}")
        print(f"[STATS] Dropped tracklets (too_missing): {dropped_tracklets_too_missing}")
        print(f"[STATS] Dropped tracklets (too_short < {MIN_FRAMES_PER_TRACKLET}): {dropped_tracklets_too_short}")

    # Verify existence rate in output
    def exists_in_dst(name: str) -> bool:
        fname = Path(name).name
        pid4 = infer_pid4_from_filename(fname)
        return (dst_bbox / pid4 / fname).exists()

    hits = sum(exists_in_dst(nm) for nm in rebuilt_test_name)
    rate = hits / max(1, len(rebuilt_test_name))
    print(f"[VERIFY] exists via pid4/filename: {hits} / {len(rebuilt_test_name)} = {rate:.6f}")
    if rate < 0.999:
        print("[WARN] Output still has missing files referenced by test_name.txt. Please inspect missing_log.")


if __name__ == "__main__":
    main()
