# Vision Sentinel — Research Notes
**Competition:** OpenCV AI Competition 2026, powered by AWS  
**Project Subtitle:** Evidence-Driven Restricted-Zone Entry Verification with OpenCV 5  
**Date of Research:** October 17, 2026  
**Document Status:** Complete & Verified  

---

## 1. Primary Sources & Research Summary

### 1.1 Official Competition Sources

| Source URL | Source Title | Date Checked | Key Findings | Supported Decisions |
| :--- | :--- | :--- | :--- | :--- |
| `https://opencv26.devpost.com/` | OpenCV AI Competition 2026 Devpost | 2026-10-17 | - Focus: Physical AI / Generative AI where visual understanding drives decisions/actions.<br>- Final deadline: October 26, 2026 at 11:59 PM PT.<br>- Cash prizes: 1st ($5,000), 2nd ($3,000), 3rd ($2,000).<br>- Special Awards: "Best Use of COOL" ($1,000) & "Agentic Vision Award" ($1,000). | Project designed squarely around the Agentic Vision criteria: dynamic tool selection where visual evidence influences the subsequent OpenCV tool call. |
| `https://opencv26.devpost.com/rules` | Official Competition Rules | 2026-10-17 | - Mandatory requirement 1: Use OpenCV 5 for substantive image/video processing.<br>- Mandatory requirement 2: Meaningful component executed or packaged for AWS.<br>- Special Award "Agentic Vision": visual evidence must influence a plan, tool call, action, or human review request. A static chatbot explaining a detection is explicitly disqualified. | Architected with a multi-step agent and deterministic policy loop. OpenCV 5 is the primary compute engine. |
| `https://opencv.org/opencv-ai-competition-2026/` | OpenCV 2026 Official Portal | 2026-10-17 | - Substantive CV required. Autonomous inspection and safety monitoring are featured tracks.<br>- OpenCV 5 DNN engine is modernized. | Focus entirely on restricted-zone verification without bloat (no generic IoT/safety dashboards). |
| `https://pypi.org/project/opencv-python-headless/` | PyPI - opencv-python-headless | 2026-10-17 | - Available version: `5.0.0.93` (official wheel built for cp37-abi3 on Windows, Linux, macOS, ARM64). | Pin `opencv-python-headless==5.0.0.93` in `requirements.txt` and verify `cv2.__version__` dynamically at startup. |
| `https://docs.opencv.org/5.x/` | OpenCV 5 Documentation | 2026-10-17 | - Modernized DNN module, background subtractors (`createBackgroundSubtractorMOG2`, `createBackgroundSubtractorKNN`), morphology, contour hierarchy, polygon intersection. | Build core motion extraction, bounding box tracking, and polygon spatial relations using tested OpenCV 5 primitives. |
| `https://opencv.org/cool/` | Cloud-Optimized OpenCV Library (COOL) | 2026-10-17 | - Available via AWS Marketplace specifically optimized for AWS Graviton (Arm Neoverse).<br>- Requires AWS Graviton instance execution and subscription. | Honesty constraint: Because unauthorized paid AWS/Marketplace spend is barred, COOL execution is not falsely claimed. Complete benchmark tooling is provided for Graviton verification when authorized. |
| `https://ai.google.dev/gemini-api/docs/models` & `pricing` | Google Gemini API Models & Pricing | 2026-10-17 | - `gemini-3.8-flash` & `gemini-3.5-flash-lite` support multimodal image understanding.<br>- Free Tier available: 15 RPM, 1M TPM, 1500 RPD under Google AI Studio.<br>- Free Tier data privacy: Google terms state submitted media on free tier may be used for product improvement. | 1. Implement Gemini STRICTLY LAST.<br>2. Default disabled (`GEMINI_ENABLED=false`).<br>3. Zero media transmitted without explicit user consent.<br>4. Never send entire videos; only send small cropped ambiguous patches.<br>5. Never automatically fall back to paid tiers. |
| `https://aws.amazon.com/free/` | AWS Free Tier Guidelines | 2026-10-17 | - EC2 750 hours/month of t2.micro/t3.micro (and t4g.micro trial in specific regions).<br>- No unauthorized resource creation without user consent. | Package container definitions (Dockerfile, docker-compose, deploy/aws/Dockerfile) and smoke tests locally. Zero unauthorized cloud charges. |

---

## 2. Licensing and Model Considerations

### 2.1 Computer Vision & Core Pipeline
- **OpenCV 5 (`opencv-python-headless==5.0.0.93`)**: Apache 2.0 license. Compatible with commercial and open-source competition submission.
- **Classical Motion Pipeline (Mode B - Default)**: Zero external model weights. 100% self-contained OpenCV background subtraction, morphological noise filtering, contour extraction, centroid association, and polygon geometry.

### 2.2 Local Object Detector (Mode A)
- **Model Candidate Evaluated**: MediaPipe Person Detection (`person_detection_mediapipe_2023mar.onnx`) from the official OpenCV Zoo (`opencv/opencv_zoo`).
- **License**: Apache 2.0 license.
- **Inference Runtime**: Standard OpenCV 5 DNN (`cv2.dnn.readNetFromONNX`). No heavy PyTorch or TensorFlow dependencies required.
- **Fallback Architecture**: If the model file is not present on disk, the system runs gracefully in **Mode B** (`moving_object_candidate`), clearly displaying the operational mode in the UI, API, and agent trace.

---

## 3. Competition Requirements Mapping

| Competition Criterion / Rule | Vision Sentinel Implementation Artifact |
| :--- | :--- |
| **Technical Execution (30%)** | Full OpenCV 5 pipeline (`app/vision/`): frame ingest, MOG2/KNN background subtraction, morphological noise reduction, tracking, polygon ray-casting and intersection, synthetic video generator with ground truth. Tested with Pytest. |
| **Innovation (20%)** | Dynamic Agentic Decision Loop (`app/agent/`): Evidence-driven policy engine selecting OpenCV tools dynamically (`inspect_roi`, `extract_foreground`, `recheck_temporal_window`, `evaluate_zone_geometry`, `verify_track_continuity`). |
| **Real-world Impact (20%)** | Addresses the real fixed-camera false-alarm problem in restricted zones (e.g., perimeter fences, server rooms, dangerous machinery zones) by differentiating turning objects, noisy shadows, and momentary occlusions from confirmed intrusions. |
| **User Experience (10%)** | Clean responsive web application (FastAPI + Vanilla CSS/JS) featuring one-click demo, polygon zone configuration, video playback with bounding boxes, real-time agent trace viewer, event timeline, and downloadable reports. |
| **Documentation & Presentation (10%)** | Exhaustive technical documentation (`docs/`): Architecture, Technical Report, Evaluation Report, Responsible Use & Privacy Policy, 5-minute Demo Video Script, Devpost Submission Copy. |
| **Cloud Delivery & Reproducibility (10%)** | Dockerfile, docker-compose.yml, AWS ECS/EC2 deployment guide in `deploy/aws/`, automated benchmark scripts, deterministic synthetic test datasets. |
| **Special Award: Agentic Vision** | Explicit tool registry (`app/agent/registry.py`) with validated schemas. Multi-step reasoning where visual evidence updates internal state machine (`OUTSIDE` -> `APPROACHING` -> `CANDIDATE` -> `NEEDS_RECHECK` -> `CONFIRMED`/`DISMISSED`/`INCONCLUSIVE`). |
| **Special Award: COOL (Arm/Graviton)** | Benchmarking script in `benchmarks/run_benchmark.py` measuring latency, throughput, and memory. Transparently flags status as ready for AWS Graviton execution when authorized. |

---

## 4. Unresolved or Environment-Specific Limitations
1. **Host Environment**: Windows host without Docker daemon installed. Container builds must be validated via syntax and configuration checks, and verified on environments with Docker.
2. **AWS Deployment Policy**: Strictly no unauthorized cloud charges. Deployment bundle is packaged and verified locally; live cloud instance creation is marked pending user cloud authorization.
3. **OpenCV 5 Headless Package**: Official `5.0.0.93` wheels are available on PyPI for `cp37-abi3-win_amd64` and Linux `manylinux_2_17`.
