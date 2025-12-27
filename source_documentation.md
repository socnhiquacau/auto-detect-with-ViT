# Documentation for the Source Code

## Overview
This document provides an overview and explanation of the source code files in the project. It includes details about the purpose of each file, key functions, and how they interact with each other.

---

## File Descriptions

### 1. `add_known_persons.py`
- **Purpose**: Handles the addition of known persons by extracting features from images in the `known_persons` directory.
- **Key Features**:
  - Processes images and groups them by person.
  - Extracts feature vectors for each image.
  - Saves the extracted features into the database.
- **Notes**:
  - Refactor to use common functions and environment variables.
  - Ensure methods are modular and not dependent on `arg[]`.

### 2. `common.py`
- **Purpose**: Contains utility functions and shared logic used across the project.
- **Key Features**:
  - Provides reusable functions for logging, configuration, and other common tasks.

### 3. `database.py`
- **Purpose**: Manages database interactions.
- **Key Features**:
  - Handles database connections and queries.
  - Provides functions for saving and retrieving data.

### 4. `feature_comparison.py`
- **Purpose**: Compares feature vectors to identify similarities between images.
- **Key Features**:
  - Implements algorithms for feature comparison.
  - Provides a class-based structure for modularity.

### 5. `feature_extractor.py`
- **Purpose**: Extracts feature vectors from images.
- **Key Features**:
  - Uses pre-trained models to generate feature vectors.
  - Supports batch processing of images.

### 6. `image_enhancement.py`
- **Purpose**: Enhances image quality for better feature extraction.
- **Key Features**:
  - Provides functions for image preprocessing.
  - Includes methods for resizing, denoising, and normalization.

### 7. `main.py`
- **Purpose**: Entry point for the application.
- **Key Features**:
  - Initializes the application.
  - Handles command-line arguments and starts the appropriate processes.

### 8. `make_mevid_subset_fixed.py`
- **Purpose**: Processes video datasets to create subsets.
- **Key Features**:
  - Extracts frames from videos.
  - Filters and organizes frames based on specific criteria.

### 9. `models.py`
- **Purpose**: Defines machine learning models used in the project.
- **Key Features**:
  - Loads pre-trained models.
  - Provides functions for inference and training.

### 10. `video_processor.py`
- **Purpose**: Handles video processing tasks.
- **Key Features**:
  - Extracts frames from videos.
  - Applies feature extraction and enhancement to video frames.

---

## Directory Structure

### `DATABASE/`
- Contains scripts and configurations for database setup and management.
- **Key Files**:
  - `setup_database.py`: Sets up the database schema.
  - `import_data.py`: Imports data into the database.

### `known_persons/`
- Stores images of known persons.
- Images are named using the format `name_number` (e.g., `hl_1.png`).

### `models/`
- Stores pre-trained models and related files.
- **Key Files**:
  - `dino_vits16_epoch100.pth`: Pre-trained model for feature extraction.
  - `yolov8_person_detection.pt`: Model for person detection.

### `output/`
- Stores output files such as detected images and logs.

### `temp/`
- Temporary directory for intermediate files.

### `uploads/`
- Directory for uploaded files.

---

## Common Issues and Debugging

### 1. **Performance Issues**
- **Symptom**: High SSD usage causing the application to hang.
- **Solution**: Optimize image processing and database interactions.

### 2. **AttributeError in `image_enhancement`**
- **Symptom**: `AttributeError: module 'image_enhancement' has no attribute 'enhance'`.
- **Solution**: Ensure the `enhance` function is implemented in `image_enhancement.py`.

### 3. **Feature Vector Errors**
- **Symptom**: `Processing failed: 'feature_vector'`.
- **Solution**: Verify the structure of the `feature_vector` and ensure it matches the expected format.

---

## Future Improvements
- Refactor code to improve modularity and readability.
- Add detailed comments in both English and Vietnamese.
- Optimize database queries and image processing pipelines.
- Implement better error handling and logging mechanisms.

---

## References
- [Project README](./README.md)
- [Refactor Guide](./REFACTOR_GUIDE.md)
