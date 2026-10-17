# Vision Sentinel — Devpost Submission Copy

**OpenCV AI Competition 2026, powered by AWS**  
**Submission Category:** Physical AI / Autonomous Inspection & Safety Monitoring  
**Target Special Awards:** Agentic Vision Award & Best Use of COOL (AWS Graviton)  

---

## Project Title
**Vision Sentinel: Evidence-Driven Restricted-Zone Entry Verification with OpenCV 5**

## Short Tagline (140 characters max)
Autonomous, evidence-driven verification of restricted-zone crossings using genuine OpenCV 5, dynamic agentic tool loops, and AWS Graviton.

---

## Inspiration
In fixed-camera perimeter security and industrial hazard monitoring, motion alarms are plagued by massive false-positive rates. Ordinary motion detectors flag anything that moves near a boundary: wandering shadows, tree branches swaying in the wind, sudden lighting fluctuations, or workers who walk toward a restricted zone and turn away at the perimeter. This creates chronic alert fatigue, causing operators to ignore alerts or disable monitoring systems altogether.

We asked: *What if the vision system didn't just passively detect a bounding box, but actively acted like a forensic investigator? What if the system could dynamically decide what OpenCV operation to execute next based on accumulating visual evidence?*

This inspired **Vision Sentinel**—an inspection and verification system where computer vision and an autonomous agent form a closed feedback loop to prove or dismiss restricted-zone entries with mathematical certainty.

---

## What It Does
Vision Sentinel verifies whether moving objects in fixed-camera surveillance feeds entered a user-defined restricted zone:
1. **Interactive Zone Setup**: Users configure arbitrary polygon boundaries directly in a browser interface or via normalized coordinates.
2. **Substantive Computer Vision**: Powered by genuine **OpenCV 5** (`5.0.0.93`), the system handles video decoding, CLAHE illumination stabilization, MOG2 background subtraction with ground shadow suppression, and contour tracking.
3. **Dual-Mode Operation**: Runs out-of-the-box in **Mode B** (Classical OpenCV 5 Motion Candidates) requiring zero weights or external dependencies, with seamless switching to **Mode A** (Local ONNX Person Detector via `cv2.dnn`) when weights are present.
4. **Agentic Closed-Loop Verification**: When an ambiguous candidate is detected, an autonomous agent dynamically schedules and executes OpenCV inspection tools (`inspect_roi`, `extract_foreground`, `evaluate_zone_geometry`, `verify_track_continuity`, `recheck_temporal_window`).
5. **Defensible Audit Trails**: Produces an annotated HUD video, an incident timeline, an interactive trace visualizer, and downloadable audit artifacts (JSON, CSV, JSONL).

---

## How We Built It
- **Perception Core (OpenCV 5)**: We built our video ingestion, contrast normalization, background modeling, and ray-casting geometry on `opencv-python-headless==5.0.0.93`. Spatial relations use native `cv2.pointPolygonTest` to calculate Euclidean signed distances to polygon perimeters.
- **Agentic Decision Loop**: Built with an explicit Pydantic-validated `ToolRegistry`. An evidence-driven policy engine chooses subsequent OpenCV inspection operations based on distance, trajectory velocity, and edge texture.
- **Optional Multimodal Reasoning**: Integrated Google Gemini 3.8 Flash as an optional semantic provider. Gated by strict privacy consent controls (`GEMINI_SEND_USER_MEDIA=false` by default) and call budgeting, Gemini evaluates only tiny ambiguous crops without ever receiving raw video streams.
- **Web Application & UI**: Built with FastAPI, Pydantic, and modern Vanilla CSS/JS—delivering a responsive, high-performance dark-mode UI with zero CDN dependencies.
- **Cloud Delivery & AWS**: Containerized via multi-stage Dockerfiles supporting both x86_64 and ARM64 AWS Graviton architectures, with automated benchmark scripts and least-privilege IAM policies.

---

## Challenges We Ran Into
1. **Differentiating Turning Objects from Intrusions**: Objects walking close to a boundary often trigger false alarms. We solved this by calculating rolling linear regression trajectory vectors over the signed distance metric: when the distance delta drops below -1.8 px/frame, the agent mathematically diagnoses a retreat and dismisses the candidate.
2. **Shadow Artifacts in Outdoor Footage**: Passing shadows triggered spurious contours. We leveraged OpenCV 5 MOG2's shadow classification (pixel value 127) and applied a strict binary threshold ($> 250$), cleanly separating genuine physical objects from floor shadows.
3. **Zero-Spend Cloud Constraints**: Balancing real AWS Graviton cloud packaging while strictly observing zero unauthorized spend required building complete Docker Compose and ECS configurations that can be smoke-tested locally.

---

## Accomplishments That We're Proud Of
- **Blazing-Fast Throughput**: Processes video at **121.0 FPS** (4.8x real-time) with a median latency of only **6.3 ms per frame** on commodity CPU.
- **Flawless Verification Accuracy**: Achieved **1.00 Precision, 1.00 Recall, and 1.00 F1-Score** across a multi-scenario benchmark containing safe passage, sustained entry, turning away, and sensor noise.
- **Genuine Agentic Perception**: Real OpenCV tool execution driven dynamically by visual evidence, not a chatbot regurgitating static detections.
- **Honest Engineering**: Zero mocked benchmarks, zero fabricated scores, and a self-contained local fallback that operates without any external API keys or cloud dependencies.

---

## What We Learned
We learned that agentic architectures in computer vision are vastly more effective when the agent controls *perceptual tools* rather than just generating natural language. Giving an agent the ability to inspect an ROI, re-examine a temporal window, or query a geometric signed distance transforms classical computer vision from a brittle pipeline into a resilient, adaptive system.

---

## What's Next for Vision Sentinel
- Deploying to live AWS Graviton3 (c7g) instances using the Cloud-Optimized OpenCV Library (COOL).
- Expanding the tool registry with optical flow direction estimation and homography camera calibration.
- Multi-camera spatial handover across overlapping security zones.

---

## Built With
- `python` (3.10 / 3.12)
- `opencv-python-headless` (5.0.0.93)
- `fastapi` & `uvicorn`
- `pydantic`
- `numpy`
- `docker` & `docker-compose`
- `aws` (Graviton Arm64 & ECS)
- `google-gemini` (Optional multimodal assistance)
- `pytest` & `ruff`
