# Vision Sentinel — Responsible Use & Security Policy

**OpenCV AI Competition 2026, powered by AWS**  
**Evidence-Driven Restricted-Zone Entry Verification with OpenCV 5**

---

## 1. Core Principles

Vision Sentinel is an evidence-gathering, verification, and audit tool designed for post-hoc or human-in-the-loop review of fixed-camera video. It is **not** an autonomous physical security or emergency dispatch system.

The application adheres strictly to the following principles:
1. **Human-in-the-Loop Review**: All safety-critical outcomes (intrusions, perimeter breaches) are flagged for human sign-off; the system initiates no automated physical actions.
2. **Privacy by Default**: Video streams and image frames remain on the local host or private deployment boundary by default.
3. **No Autonomous Actuation**: The agent has no access to door locks, barrier controls, alarm sirens, or external API relays.
4. **Transparent Audit Trails**: Every decision is accompanied by raw geometric evidence, OpenCV tool outputs, and structured JSONL logs.

---

## 2. Privacy & Data Handling Policy

### 2.1 Local Processing Guarantee
- By default, all video decoding, background subtraction, tracking, polygon ray-casting, and agentic tool executions occur 100% locally on the host CPU.
- Uploaded media and analysis artifacts are stored in a dedicated local directory (`./data/uploads/` and `./data/outputs/`).

### 2.2 Optional Multimodal Cloud Inference (Google Gemini)
When the optional Google Gemini multimodal provider is enabled:
- **Explicit Consent Gate**: The system enforces `GEMINI_SEND_USER_MEDIA=false` by default. Real user media is blocked from cloud transmission unless explicit opt-in consent is provided in `.env` or during job initiation.
- **Crop-Only Transmission**: Under no circumstance is a full video or raw video stream transmitted to cloud endpoints. Only tiny, ambiguous image patches (crops of the specific candidate bounding box) are transmitted for semantic clarification.
- **Free-Tier Terms Transparency**: Google Gemini Free Tier documentation indicates that submitted prompt data may be used to improve Google products. Users handling sensitive, proprietary, or personally identifiable surveillance footage must keep `GEMINI_ENABLED=false` or ensure `GEMINI_SEND_USER_MEDIA=false` to guarantee zero data egress.
- **Secret Hygiene**: API keys are loaded strictly from environment variables, never hard-coded, never logged to stdout/stderr, and never included in API responses or browser JavaScript.

---

## 3. Application Security & Hardening

Vision Sentinel enforces rigorous defense-in-depth measures against common web application and computer vision vulnerabilities:

### 3.1 Input Validation & Path Traversal Mitigation
- **Strict Boundary Resolution**: All artifact download and file viewing endpoints resolve requested paths against `settings.data_dir.resolve()` using `Path.is_relative_to()`. Any attempt to access system files (e.g. `/etc/passwd` or `..\..\Windows\win.ini`) results in an HTTP 404 response.
- **Server-Generated Filenames**: Ingested files are assigned UUID4-based filenames to prevent directory traversal or file-system poisoning via malicious uploaded filenames.
- **Media Validation**: File headers and video container structures are inspected via `cv2.VideoCapture` before processing. Malformed, truncated, or zero-byte files are rejected before allocation.

### 3.2 Resource Limits & DoS Prevention
- **Max File Size**: 100 MB maximum per upload (`MAX_UPLOAD_MB`).
- **Max Video Duration**: 180 seconds maximum processing window (`MAX_VIDEO_SECONDS`).
- **Max Resolution**: 1920 pixels maximum dimension; larger frames are scaled to prevent memory exhaustion.
- **Bounded Agent Execution**: The autonomous agent enforces a strict budget of maximum 5 steps per candidate, preventing infinite feedback cycles.

### 3.3 Security Guardrails on Agent Tools
- **Sandboxed Tool Registry**: Agent tools are strictly enumerated in `app/agent/registry.py`.
- **Zero Shell / Eval Access**: Tools execute pre-compiled OpenCV functions with validated Pydantic parameters. The agent cannot execute arbitrary Python code, invoke shell commands, or modify files.
- **Output Sanitization**: External model recommendations are validated against controlled enums (`RecommendedInspection`) and can be discarded by deterministic policy if inconsistent with measured geometric facts.

---

## 4. Operational Limitations & Honest Disclosures

Vision Sentinel provides probabilistic and geometric evidence, not absolute ground truth:
1. **Camera Geometry**: Accuracy depends on fixed camera placement. Strong perspective distortion, extreme wide-angle fish-eye effects, or low viewing angles may degrade polygon distance calculations unless camera calibration is applied.
2. **Severe Environmental Occlusion**: Heavy fog, torrential rain, direct optical glare into the lens, or physical obstructions blocking more than 70% of a candidate over multiple frames may lead to `INCONCLUSIVE` classifications.
3. **No Safety-Critical Certification**: Vision Sentinel is **not certified** under IEC 61508, ISO 13849, or UL 2017 for life-safety or industrial interlock applications. It must not be used as the sole safety measure preventing machinery operation or hazard access.
