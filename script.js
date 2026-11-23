const dropzone = document.getElementById("dropzone");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const fileInput = document.getElementById("fileInput");
const fileNameBadge = document.getElementById("fileNameBadge");

const btnChooseFile = document.getElementById("btnChooseFile");
const btnClear = document.getElementById("btnClear");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnAnalyzeText = document.getElementById("btnAnalyzeText");

const toggleKeypoints = document.getElementById("toggleKeypoints");

const resultCard = document.getElementById("resultCard");
const resStatus = document.getElementById("resStatus");
const resConf = document.getElementById("resConf");
const resKpts = document.getElementById("resKpts");

const SKELETON_PAIRS = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
];

let rawImageDataUrl = null;
let rawImageElement = null;
let lastDetections = null;
let lastBackendAnnotatedImage = null;
let hasAnalyzed = false;

function resetStats() {
  hasAnalyzed = false;
  lastDetections = null;
  lastBackendAnnotatedImage = null;
  resultCard.hidden = true;
  resStatus.textContent = "-";
  resConf.textContent = "0%";
  resKpts.textContent = "0 / 17";
}

function showPreviewUI(label) {
  uploadPrompt.hidden = true;
  previewContainer.hidden = false;
  fileNameBadge.textContent = label;
}

function showUploadPromptUI() {
  rawImageDataUrl = null;
  rawImageElement = null;
  previewImage.removeAttribute("src");
  previewContainer.hidden = true;
  uploadPrompt.hidden = false;
  btnAnalyzeText.textContent = "Analisis Gerakan dengan YOLOv11-Pose";
  resetStats();
}

function countValidKeypoints(detection) {
  if (!detection) return 0;
  if (typeof detection.keypoints_detected === "number") {
    return detection.keypoints_detected;
  }
  if (Array.isArray(detection.keypoints)) {
    return detection.keypoints.filter((k) => k.valid || (k.conf > 0.25 && k.x > 0 && k.y > 0)).length;
  }
  return 0;
}

function renderVisualByToggles() {
  if (!hasAnalyzed || !rawImageDataUrl) return;

  const showPose = toggleKeypoints.checked;

  const hasCoords =
    rawImageElement &&
    Array.isArray(lastDetections) &&
    lastDetections.length > 0 &&
    lastDetections.some((d) => Array.isArray(d.keypoints) && Array.isArray(d.box));

  if (hasCoords) {
    const canvas = document.createElement("canvas");
    const w = rawImageElement.naturalWidth || rawImageElement.width;
    const h = rawImageElement.naturalHeight || rawImageElement.height;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(rawImageElement, 0, 0, w, h);

    const scale = Math.max(1, Math.min(w, h) / 640);
    const colorBlue = "#1a56db";
    const colorWhite = "#ffffff";

    for (const det of lastDetections) {
      const [x1, y1, x2, y2] = det.box;
      const label = (det.label || "").toUpperCase();
      const confPct = Math.floor((det.confidence || 0) * 100 + 1e-6);
      const tag = `${label} ${confPct}%`;

      ctx.strokeStyle = colorBlue;
      ctx.lineWidth = Math.max(2, Math.round(2.5 * scale));
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const fontSize = Math.max(14, Math.round(16 * scale));
      ctx.font = `700 ${fontSize}px Inter, sans-serif`;
      const textMetrics = ctx.measureText(tag);
      const padX = Math.round(8 * scale);
      const padY = Math.round(6 * scale);
      const tagH = fontSize + padY * 2;
      const tagW = textMetrics.width + padX * 2;
      const tagY = Math.max(0, y1 - tagH);

      ctx.fillStyle = colorBlue;
      ctx.fillRect(x1, tagY, tagW, tagH);

      ctx.fillStyle = colorWhite;
      ctx.textBaseline = "middle";
      ctx.fillText(tag, x1 + padX, tagY + tagH / 2);

      const pts = det.keypoints || [];

      if (showPose) {
        ctx.lineCap = "round";
        for (const [a, b] of SKELETON_PAIRS) {
          if (a < pts.length && b < pts.length && pts[a].valid && pts[b].valid) {
            ctx.beginPath();
            ctx.moveTo(pts[a].x, pts[a].y);
            ctx.lineTo(pts[b].x, pts[b].y);
            ctx.strokeStyle = colorWhite;
            ctx.lineWidth = Math.max(4, Math.round(4.5 * scale));
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(pts[a].x, pts[a].y);
            ctx.lineTo(pts[b].x, pts[b].y);
            ctx.strokeStyle = colorBlue;
            ctx.lineWidth = Math.max(2, Math.round(2.5 * scale));
            ctx.stroke();
          }
        }

        const rOuter = Math.max(5, Math.round(6 * scale));
        const rInner = Math.max(3, Math.round(4 * scale));
        for (const pt of pts) {
          if (pt.valid) {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, rOuter, 0, Math.PI * 2);
            ctx.fillStyle = colorWhite;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, rInner, 0, Math.PI * 2);
            ctx.fillStyle = colorBlue;
            ctx.fill();
          }
        }
      }
    }

    previewImage.src = canvas.toDataURL("image/jpeg", 0.92);
    return;
  }

  // Fallback jika server belum di-restart
  if (!showPose) {
    previewImage.src = rawImageDataUrl;
  } else if (lastBackendAnnotatedImage) {
    previewImage.src = lastBackendAnnotatedImage;
  }
}

function applyDetectionResult(data) {
  resultCard.hidden = false;
  hasAnalyzed = true;
  lastBackendAnnotatedImage = data.image || null;
  lastDetections = Array.isArray(data.detections) ? data.detections : (data.primary ? [data.primary] : []);

  const p = data.primary || lastDetections[0] || null;
  if (!p) {
    previewImage.src = rawImageDataUrl;
    resStatus.textContent = "Tidak Terdeteksi";
    resConf.textContent = "0%";
    resKpts.textContent = "0 / 17";
    return;
  }

  let detectedCount = countValidKeypoints(p);
  if (detectedCount === 0 && p.elbow_angle !== null && p.elbow_angle !== undefined) {
    detectedCount = 17;
  }

  resStatus.textContent = (p.label || "-").toUpperCase();
  resConf.textContent = `${Math.floor((p.confidence || 0) * 100 + 1e-6)}%`;
  resKpts.textContent = `${detectedCount} / 17`;

  renderVisualByToggles();
}

async function detectImage() {
  if (!rawImageDataUrl) return;

  btnAnalyze.disabled = true;
  btnAnalyzeText.textContent = "Menganalisis Gerakan...";

  try {
    const response = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: rawImageDataUrl,
        conf: 0.25,
        show_keypoints: toggleKeypoints.checked,
        show_skeleton: toggleKeypoints.checked
      })
    });

    if (response.ok) {
      const result = await response.json();
      applyDetectionResult(result);
    }
  } finally {
    btnAnalyze.disabled = false;
    btnAnalyzeText.textContent = "Analisis Gerakan dengan YOLOv11-Pose";
  }
}

function handleSelectedFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  resetStats();

  const reader = new FileReader();
  reader.onload = () => {
    rawImageDataUrl = reader.result;
    const img = new Image();
    img.onload = () => {
      rawImageElement = img;
    };
    img.src = rawImageDataUrl;
    previewImage.src = rawImageDataUrl;
    showPreviewUI(file.name);
  };
  reader.readAsDataURL(file);
}

btnChooseFile.addEventListener("click", () => {
  fileInput.value = "";
  fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  handleSelectedFile(e.target.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleSelectedFile(e.dataTransfer.files[0]);
  }
});

btnClear.addEventListener("click", () => {
  showUploadPromptUI();
});

btnAnalyze.addEventListener("click", () => {
  if (!rawImageDataUrl) {
    fileInput.value = "";
    fileInput.click();
    return;
  }
  detectImage();
});

toggleKeypoints.addEventListener("change", () => {
  renderVisualByToggles();
});
