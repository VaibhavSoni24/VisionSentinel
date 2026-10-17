/**
 * Vision Sentinel — Client-side Application Logic
 * No external frameworks, zero CDNs, pure Vanilla JS.
 */

// Global state
const appState = {
  zonePolygon: [
    [0.52, 0.22],
    [0.92, 0.22],
    [0.92, 0.88],
    [0.52, 0.88],
  ],
  currentJobId: null,
  activeTab: "tab-analysis",
};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCanvas();
  fetchSystemStatus();
  setupActionListeners();
});

function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-target");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("badge-opencv").textContent = `OpenCV ${data.opencv_version}`;
    document.getElementById("badge-mode").textContent = data.operating_mode;
    if (data.operating_mode.includes("Mode A")) {
      document.getElementById("badge-mode").classList.add("active");
    }

    const geminiBadge = document.getElementById("badge-gemini");
    if (data.gemini_active) {
      geminiBadge.textContent = `Gemini: Active (${data.gemini_model})`;
      geminiBadge.classList.add("active");
    } else {
      geminiBadge.textContent = "Gemini: Disabled (Local Rule Fallback)";
    }
  } catch (err) {
    console.error("Failed to fetch system status:", err);
  }
}

function initCanvas() {
  const canvas = document.getElementById("zoneCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background guide grid
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw polygon
    const pts = appState.zonePolygon.map((p) => ({
      x: p[0] * canvas.width,
      y: p[1] * canvas.height,
    }));

    if (pts.length > 0) {
      ctx.fillStyle = "rgba(6, 182, 212, 0.20)";
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Draw vertices
      pts.forEach((pt, i) => {
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "10px monospace";
        ctx.fillText(`V${i + 1}`, pt.x + 8, pt.y - 4);
      });
    }

    // Update coordinate text box
    const coordEl = document.getElementById("zoneCoordsText");
    if (coordEl) {
      coordEl.value = JSON.stringify(appState.zonePolygon, null, 2);
    }
  }

  draw();

  const updateBtn = document.getElementById("btnUpdateZone");
  if (updateBtn) {
    updateBtn.addEventListener("click", () => {
      try {
        const text = document.getElementById("zoneCoordsText").value;
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed) && parsed.length >= 3) {
          appState.zonePolygon = parsed;
          draw();
        } else {
          alert("Polygon must contain at least 3 [x, y] coordinates in [0.0, 1.0].");
        }
      } catch (e) {
        alert("Invalid JSON format: " + e.message);
      }
    });
  }

  const resetBtn = document.getElementById("btnResetZone");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      appState.zonePolygon = [
        [0.52, 0.22],
        [0.92, 0.22],
        [0.92, 0.88],
        [0.52, 0.88],
      ];
      draw();
    });
  }
}

function setupActionListeners() {
  const btnRunDemo = document.getElementById("btnRunDemo");
  if (btnRunDemo) {
    btnRunDemo.addEventListener("click", runDemo);
  }

  const uploadForm = document.getElementById("uploadForm");
  if (uploadForm) {
    uploadForm.addEventListener("submit", handleUpload);
  }
}

function setProgress(show, text = "") {
  const pContainer = document.getElementById("progressContainer");
  const pBar = document.getElementById("progressBar");
  const statusMsg = document.getElementById("statusMessage");

  if (show) {
    pContainer.style.display = "block";
    pBar.style.width = "45%";
    statusMsg.textContent = text;
  } else {
    pBar.style.width = "100%";
    setTimeout(() => {
      pContainer.style.display = "none";
      pBar.style.width = "0%";
      statusMsg.textContent = "";
    }, 400);
  }
}

async function runDemo() {
  const btn = document.getElementById("btnRunDemo");
  btn.disabled = true;
  btn.textContent = "Processing Demo...";
  setProgress(true, "Executing OpenCV 5 synthetic demo scenario...");

  const formData = new FormData();
  formData.append("zone_points_json", JSON.stringify(appState.zonePolygon));

  try {
    const res = await fetch("/api/demo", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await res.json();
    populateResults(data);
  } catch (err) {
    alert("Demo analysis failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Built-in Synthetic Demo";
    setProgress(false);
  }
}

async function handleUpload(e) {
  e.preventDefault();
  const fileInput = document.getElementById("videoFile");
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Please select a video file first.");
    return;
  }

  const file = fileInput.files[0];
  const btn = document.getElementById("btnUploadSubmit");
  btn.disabled = true;
  btn.textContent = "Uploading & Analyzing...";
  setProgress(true, "Analyzing video frames with OpenCV 5...");

  const formData = new FormData();
  formData.append("video", file);
  formData.append("zone_points_json", JSON.stringify(appState.zonePolygon));

  const consentCb = document.getElementById("userConsentCheckbox");
  if (consentCb) {
    formData.append("send_user_media", consentCb.checked);
  }

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Pipeline processing error");
    }

    const data = await res.json();
    populateResults(data);
  } catch (err) {
    alert("Analysis failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Upload & Analyze Video";
    setProgress(false);
  }
}

function populateResults(result) {
  appState.currentJobId = result.job_id;

  // 1. Video Player
  const videoPlayer = document.getElementById("resultVideo");
  const videoSource = document.getElementById("videoSource");
  if (videoPlayer && videoSource) {
    videoSource.src = `/api/download/${result.job_id}/video`;
    videoPlayer.load();
    videoPlayer.style.display = "block";
    document.getElementById("zoneCanvas").style.display = "none";
  }

  // 2. Incident Summary Cards
  const incidentsList = document.getElementById("incidentsList");
  if (incidentsList) {
    incidentsList.innerHTML = "";
    if (result.incidents.length === 0) {
      incidentsList.innerHTML = "<p class='text-dim'>No movement candidates detected.</p>";
    } else {
      result.incidents.forEach((inc) => {
        const item = document.createElement("div");
        item.className = "card";
        item.style.padding = "0.75rem";
        item.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong>Candidate ID: ${inc.candidate_id} (${inc.category})</strong>
            <span class="outcome-badge outcome-${inc.final_state}">${inc.final_state}</span>
          </div>
          <div style="font-size:0.8rem; color:var(--text-muted);">
            Observed: ${inc.frames_observed} frames | Evidence Score: <strong>${inc.evidence_score}</strong>
            ${inc.human_review_required ? " | <span style='color:var(--status-confirmed);'>Review Required</span>" : ""}
          </div>
          <p style="font-size:0.85rem; margin-top:0.2rem;">${inc.rationale}</p>
        `;
        incidentsList.appendChild(item);
      });
    }
  }

  // 3. Timeline Table
  const timelineTbody = document.getElementById("timelineTbody");
  if (timelineTbody) {
    timelineTbody.innerHTML = "";
    result.timeline.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.timestamp.toFixed(2)}s</td>
        <td>Frame ${item.frame_index}</td>
        <td>ID ${item.candidate_id}</td>
        <td><span class="outcome-badge outcome-${item.state}">${item.state}</span></td>
        <td>${item.evidence_score.toFixed(2)}</td>
        <td>${item.message}</td>
      `;
      timelineTbody.appendChild(tr);
    });
  }

  // 4. Agent Trace Cards
  const traceFeed = document.getElementById("traceFeed");
  if (traceFeed) {
    traceFeed.innerHTML = "";
    if (result.trace_steps.length === 0) {
      traceFeed.innerHTML = "<p class='text-dim'>No agent steps logged for this run.</p>";
    } else {
      result.trace_steps.forEach((step) => {
        const card = document.createElement("div");
        card.className = "trace-card";
        card.innerHTML = `
          <div class="trace-header">
            <span>STEP ${step.step_index} | Candidate ${step.candidate_id} @ ${step.timestamp}s (Frame ${step.frame_index})</span>
            <span class="outcome-badge outcome-${step.new_state}">${step.new_state}</span>
          </div>
          <div class="trace-body">
            <strong>Tool Invoked:</strong> <span class="trace-tool">${step.chosen_tool}</span>
            <span class="trace-meta"> | Policy Condition: ${step.policy_condition}</span>
          </div>
          <div class="trace-code">
            <strong>Args:</strong> ${JSON.stringify(step.tool_arguments)}<br>
            <strong>Result:</strong> ${JSON.stringify(step.tool_result)}
          </div>
          <div style="font-size:0.8rem; color:var(--text-muted);">
            <strong>Rationale:</strong> ${step.rationale} | Evidence Score: <strong>${step.updated_evidence_score}</strong>
          </div>
        `;
        traceFeed.appendChild(card);
      });
    }
  }

  // 5. Update Download Links
  document.getElementById("btnDownloadVideo").href = `/api/download/${result.job_id}/video`;
  document.getElementById("btnDownloadIncidents").href = `/api/download/${result.job_id}/incidents`;
  document.getElementById("btnDownloadTimeline").href = `/api/download/${result.job_id}/timeline`;
  document.getElementById("btnDownloadTrace").href = `/api/download/${result.job_id}/trace`;
  document.getElementById("btnDownloadReport").href = `/api/download/${result.job_id}/report`;

  // Enable download buttons
  document.querySelectorAll(".dl-btn").forEach((b) => (b.style.pointerEvents = "auto"));
}
