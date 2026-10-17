# Third-Party Notices & Attribution

Vision Sentinel incorporates and links against third-party open-source software and model architectures. This document provides notices, licensing summaries, and original sources.

---

## 1. Computer Vision & Core Libraries

### OpenCV 5 (`opencv-python-headless`)
* **Project**: Open Source Computer Vision Library (OpenCV 5.x)
* **Website**: https://opencv.org/
* **License**: Apache License 2.0
* **Copyright**: (c) 2000-2026, OpenCV Team
* **Usage**: Core video acquisition, spatial analysis, MOG2/KNN background subtraction, morphological noise reduction, contour tracking, crop extraction, and DNN execution.

### NumPy
* **Project**: Fundamental package for array computing with Python
* **Website**: https://numpy.org/
* **License**: BSD 3-Clause License
* **Copyright**: (c) 2005-2026, NumPy Developers
* **Usage**: Numerical array manipulation, geometric coordinate math, matrix transformation.

---

## 2. Web Framework & Utilities

### FastAPI
* **Website**: https://fastapi.tiangolo.com/
* **License**: MIT License
* **Copyright**: (c) 2018-2026, Sebastián Ramírez

### Pydantic
* **Website**: https://docs.pydantic.dev/
* **License**: MIT License
* **Copyright**: (c) 2017-2026, Samuel Colvin

### Uvicorn
* **Website**: https://www.uvicorn.org/
* **License**: BSD 3-Clause License
* **Copyright**: (c) 2017-2026, Encode OSS Ltd.

### Python-dotenv
* **Website**: https://github.com/theskumar/python-dotenv
* **License**: BSD 3-Clause License
* **Copyright**: (c) 2014-2026, Saurabh Kumar

### Jinja2
* **Website**: https://palletsprojects.com/p/jinja/
* **License**: BSD 3-Clause License
* **Copyright**: (c) 2007-2026, Pallets

### HTTPX
* **Website**: https://www.encode.io/httpx/
* **License**: BSD 3-Clause License
* **Copyright**: (c) 2019-2026, Encode OSS Ltd.

---

## 3. Pretrained Model Architecture & Weights

### MediaPipe Person Detection (OpenCV Zoo)
* **Project**: OpenCV Zoo (`person_detection_mediapipe`)
* **Source**: https://github.com/opencv/opencv_zoo & https://huggingface.co/opencv/person_detection_mediapipe
* **Architecture**: BlazePose / MediaPipe upper-body & full-body detector
* **License**: Apache License 2.0
* **Copyright**: (c) Google LLC and OpenCV Team
* **Status**: Optional local detector (Mode A). Downloadable via `scripts/setup_model.py`. When absent from the filesystem, Vision Sentinel seamlessly executes Mode B (Classical OpenCV Motion Candidate) with zero external weights.

---

## 4. Optional Cloud & Foundation Model Services

### Google GenAI SDK & Gemini
* **Project**: `google-genai` Python SDK
* **Documentation**: https://ai.google.dev/gemini-api/docs
* **License**: Apache License 2.0
* **Copyright**: (c) Google LLC
* **Data Notice**: Under Google AI Studio Free Tier terms, submitted media may be used to improve Google products. Vision Sentinel enforces explicit per-job user consent (`GEMINI_SEND_USER_MEDIA=true`) before transmitting any cropped ROI patch to Gemini.
