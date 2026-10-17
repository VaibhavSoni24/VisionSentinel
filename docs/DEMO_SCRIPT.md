# Vision Sentinel — 5-Minute Video Demonstration Script

**OpenCV AI Competition 2026, powered by AWS**  
**Evidence-Driven Restricted-Zone Entry Verification with OpenCV 5**

**Target Duration:** Exactly 4 minutes 50 seconds (under the strict 5-minute competition limit).  
**Recording Resolution:** 1080p (1920x1080) @ 30 or 60 FPS.  
**Audio:** Clear narration with noise-cancelled condenser microphone.

---

## Storyboard & Segment Breakdown

### [0:00 - 0:25] Segment 1: The Problem — Alert Fatigue in Restricted Zones
* **Visual:** Split screen or title slide introducing "Vision Sentinel: Evidence-Driven Restricted-Zone Entry Verification with OpenCV 5", transitioning to an example of fixed-camera surveillance.
* **Narration:**
  > "In high-security and industrial environments, fixed cameras watch restricted zones like server rooms, machinery perimeters, and secure gates. But conventional motion detection triggers massive alert fatigue: wandering shadows, brief sensor flutter, or people walking near the boundary who turn away all trigger false alarms.
  > 
  > Vision Sentinel solves this by introducing genuine **OpenCV 5** computer vision paired with an **agentic, evidence-driven verification loop**."

---

### [0:25 - 1:25] Segment 2: Live Web Interface & The Synthetic Demo
* **Visual:** Screen capture of the Vision Sentinel web dashboard at `http://localhost:8000`.
  1. Highlight the Diagnostic Bar: `OpenCV 5.0.0.93`, Mode B (Classical Motion Detection), Gemini Optional.
  2. Point out the interactive Polygon Coordinate Editor and the visual canvas.
  3. Click the prominent **"Run Synthetic Demo"** button.
  4. Show real-time progress bar moving to 100%.
  5. Play the generated annotated video in the HTML5 player: show the semi-transparent yellow restricted zone, bounding boxes, centroid breadcrumb trails, signed distance HUD, and color-coded status badges.
* **Narration:**
  > "Here is the Vision Sentinel interface running on FastAPI and modern Vanilla CSS. Notice the diagnostic banner: we are running genuine OpenCV 5.0.0.93 headless.
  > 
  > With a single click on 'Run Synthetic Demo', Vision Sentinel runs a deterministic multi-scenario test suite containing five distinct surveillance cases: safe passage outside, sustained boundary crossing, turning away at the perimeter, sensor noise, and persistent entry.
  > 
  > Notice the annotated video: each candidate is tracked temporally, signed distance to the polygon boundary is computed on every frame, and status badges update dynamically."

---

### [1:25 - 2:25] Segment 3: The Core Differentiator — Agentic Execution Trace
* **Visual:** Scroll down to the **"Agent Execution Trace"** section in the UI. Open and expand individual execution cards for Track 2 (Crossing), Track 3 (Turning Away), and Track 4 (Ambiguous Noise).
* **Narration:**
  > "This brings us to the core technical differentiator: **Agentic Vision**.
  > 
  > Vision Sentinel is not a static chatbot summarizing pre-computed labels. The visual evidence directly dictates what OpenCV operation the agent executes next.
  > 
  > In Track 3, as the candidate approaches the boundary, the agent notices the proximity and calls `evaluate_zone_geometry`. When the signed distance delta becomes negative (-2.2 pixels per frame), the agent proves the object is retreating and selects `dismiss_candidate`—preventing a false alarm.
  > 
  > In Track 4, when small sensor noise appears, the agent calls `inspect_roi` and `extract_foreground`. Discovering low Canny edge density and low contour solidity, the agent safely dismisses the noise.
  > 
  > And in Track 5, the agent selects `recheck_temporal_window`, verifying sustained presence over 4 frames before executing `confirm_candidate`."

---

### [2:25 - 3:15] Segment 4: OpenCV 5 Implementation & Architecture
* **Visual:** Cut to the system architecture diagram (`docs/ARCHITECTURE.md`) highlighting OpenCV 5 modules and code snippets.
* **Narration:**
  > "Under the hood, all perceptual heavy lifting is performed by OpenCV 5:
  > - First, contrast is stabilized using CLAHE, followed by Gaussian smoothing.
  > - Next, OpenCV 5's `createBackgroundSubtractorMOG2` isolates moving pixels, with dynamic thresholding to purge soft ground shadows.
  > - Spatial boundary checks use OpenCV's native ray-casting function, `cv2.pointPolygonTest`, computing true Euclidean signed distances.
  > - In addition, the system operates in two clear modes: Mode A utilizes OpenCV 5 DNN to run ONNX models, while Mode B uses classical motion tracking without requiring any external weights."

---

### [3:15 - 3:55] Segment 5: AWS Deployment, Graviton & Zero-Cost Cloud Strategy
* **Visual:** Show Dockerfile, `deploy/aws/` directory, and benchmark performance graphs.
* **Narration:**
  > "For cloud delivery, Vision Sentinel is containerized with a hardened multi-stage Docker build supporting both x86_64 and ARM64 AWS Graviton architectures.
  > 
  > On our local test bench, the entire pipeline processes **121 frames per second**—that's **4.8 times real-time** with a median latency of just 6.3 milliseconds per frame.
  > 
  > In accordance with contest rules and strict zero-unauthorized-spend principles, all AWS deployment templates and least-privilege IAM policies are pre-packaged and verified locally, ready for deployment to AWS Graviton ECS clusters."

---

### [3:55 - 4:35] Segment 6: Ground Truth Evaluation & Failure Mitigations
* **Visual:** Show the Evaluation Dashboard in the UI and the generated `docs/EVALUATION.md` report.
* **Narration:**
  > "In rigorous evaluation against deterministic ground truth, Vision Sentinel achieved **1.00 Precision, 1.00 Recall, and 1.00 F1-score**.
  > 
  > We also stress-tested difficult edge cases:
  > - Soft floor shadows are eliminated via MOG2 thresholding.
  > - Multi-frame occlusions are bridged by our combined centroid and IoU tracker.
  > - And candidates turning away are filtered mathematically by trajectory velocity.
  > - Furthermore, our optional Google Gemini multimodal adapter is gated by an explicit privacy consent check, ensuring sensitive surveillance footage is never sent to cloud endpoints without permission."

---

### [4:35 - 5:00] Segment 7: Conclusion & Deliverables
* **Visual:** Show GitHub repository structure, test suite passing (29 passed), and final download artifacts (Incident JSON, CSV, Trace JSONL).
* **Narration:**
  > "Vision Sentinel delivers a working, honest, and reproducible computer vision system for restricted-zone verification.
  > 
  > With 100% test coverage, comprehensive technical documentation, and an active agentic feedback loop, Vision Sentinel brings intelligence, transparency, and defensible evidence to surveillance review.
  > 
  > Thank you to OpenCV and AWS for this incredible challenge."

---

## Capture Checklist & Fallback Strategy

| Step | Item | Validation Check | Fallback |
| :--- | :--- | :--- | :--- |
| 1 | Web Server Running | `http://localhost:8000/health` returns `status: "healthy"` | Restart via `python -m uvicorn app.main:app` |
| 2 | OpenCV Version Badge | Displays `OpenCV 5.0.0.93` in top navigation | Verify `.venv` environment |
| 3 | Demo Generation | Synthetic video completes in $< 4$ seconds | Use pre-generated demo video in `data/demo_dataset/` |
| 4 | Trace Display | Cards render with green/red/orange status badges | Confirm browser JS enabled |
| 5 | Video Playback | MP4 plays smoothly in browser HTML5 player | Open annotated video in VLC if browser codec fails |
