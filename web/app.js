const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const imageInput = $("#imageInput");
const dropZone = $("#dropZone");
const previewImage = $("#previewImage");
const predictButton = $("#predictButton");
const clearButton = $("#clearButton");
const fileName = $("#fileName");
const fileMeta = $("#fileMeta");
const resultPanel = $(".result-panel");
const resultTitle = $("#resultTitle");
const resultSummary = $("#resultSummary");
const riskPill = $("#riskPill");
const normalValue = $("#normalValue");
const pneumoniaValue = $("#pneumoniaValue");
const normalBar = $("#normalBar");
const pneumoniaBar = $("#pneumoniaBar");
const thresholdValue = $("#thresholdValue");
const distanceValue = $("#distanceValue");
const confidenceState = $("#confidenceState");
const statusValue = $("#statusValue");
const modelName = $("#heroModel");
const modelCardName = $("#modelCardName");
const serviceStatus = $("#serviceStatus");
const serviceText = $("#serviceText");
const thresholdLine = $("#thresholdLine");
const thresholdTag = $("#thresholdTag");
const scoreMarker = $("#scoreMarker");
const scoreTag = $("#scoreTag");
const sampleButtons = $$("[data-sample]");

let selectedFile = null;
let objectUrl = null;
let serviceThreshold = 0.5;

const MODEL_DISPLAY_NAMES = {
  "torchvision:vit_b_16": "ViT-B/16",
  vit_b16: "ViT-B/16",
  "torchvision:densenet121": "DenseNet-121",
  densenet121: "DenseNet-121",
  "torchvision:convnext_tiny": "ConvNeXt-Tiny",
  convnext_tiny: "ConvNeXt-Tiny",
};

const SAMPLE_LABELS = {
  normal: "NORMAL示例",
  pneumonia: "PNEUMONIA示例",
  boundary: "阈值附近示例",
};

function displayModelName(value) {
  if (typeof value !== "string" || !value.trim()) return "模型未知";
  return MODEL_DISPLAY_NAMES[value] || value;
}

function formatPercent(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "--";
}

function clamp(value, minimum = 0, maximum = 1) {
  return Math.max(minimum, Math.min(maximum, value));
}

function setServiceState(state, text) {
  serviceStatus.classList.remove("online", "offline");
  if (state) serviceStatus.classList.add(state);
  serviceText.textContent = text;
}

function updateThresholdDisplay() {
  const percent = clamp(serviceThreshold) * 100;
  thresholdValue.textContent = serviceThreshold.toFixed(2);
  thresholdLine.style.left = `${percent}%`;
  thresholdTag.textContent = `阈值 ${percent.toFixed(0)}%`;
}

async function loadHealth() {
  try {
    const response = await fetch("health");
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error("health check failed");
    const displayName = displayModelName(data.model);
    modelName.textContent = displayName;
    modelCardName.textContent = displayName;
    if (typeof data.threshold === "number" && Number.isFinite(data.threshold)) {
      serviceThreshold = clamp(data.threshold);
      updateThresholdDisplay();
    }
    setServiceState("online", `${displayName} 已连接`);
  } catch (_error) {
    setServiceState("offline", "模型服务未连接");
  }
}

function setPanelState(state = "") {
  resultPanel.classList.remove(
    "normal-state",
    "pneumonia-state",
    "boundary-state",
    "error-state",
  );
  if (state) resultPanel.classList.add(state);
}

function resetResult() {
  resultTitle.textContent = "尚未分类";
  resultSummary.textContent = "运行后将显示两个训练标签的模型分数与阈值位置。";
  riskPill.textContent = "等待推理";
  statusValue.textContent = "未运行";
  normalValue.textContent = "--";
  pneumoniaValue.textContent = "--";
  normalBar.style.width = "0%";
  pneumoniaBar.style.width = "0%";
  distanceValue.textContent = "--";
  confidenceState.textContent = "未运行";
  scoreTag.textContent = "--";
  scoreMarker.classList.remove("visible");
  setPanelState();
  updateThresholdDisplay();
}

function setActiveSample(sampleName = "") {
  sampleButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.sample === sampleName);
  });
}

function setSelectedFile(file, label = file?.name, sampleName = "") {
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (file.type && !allowed.includes(file.type)) {
    showError("仅支持 JPG、PNG 或 WebP 图片。", "文件格式不支持");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError("图片不能超过 16 MB。", "文件过大");
    return;
  }
  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  fileName.textContent = label || file.name;
  fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  predictButton.disabled = false;
  setActiveSample(sampleName);
  resetResult();
}

async function loadSample(sampleName, runAfterLoad = true) {
  const button = sampleButtons.find((item) => item.dataset.sample === sampleName);
  if (button) button.disabled = true;
  try {
    const response = await fetch(`sample-xray/${sampleName}`);
    if (!response.ok) throw new Error("示例图片不可用");
    const blob = await response.blob();
    const extension = blob.type.includes("png") ? "png" : "jpeg";
    const file = new File([blob], `${sampleName}.${extension}`, {
      type: blob.type || "image/jpeg",
    });
    setSelectedFile(file, SAMPLE_LABELS[sampleName], sampleName);
    if (runAfterLoad) await runPrediction();
  } catch (error) {
    showError(error.message || "无法读取示例图片。", "示例加载失败");
  } finally {
    if (button) button.disabled = false;
  }
}

function outputDistanceState(score, threshold) {
  const distance = Math.abs(score - threshold);
  if (distance <= 0.08) return { label: "接近阈值", state: "boundary-state" };
  if (distance <= 0.2) return { label: "中等距离", state: score >= threshold ? "pneumonia-state" : "normal-state" };
  return { label: "远离阈值", state: score >= threshold ? "pneumonia-state" : "normal-state" };
}

function renderResult(data) {
  const normal = Number(data.normal_probability);
  const pneumonia = Number(data.pneumonia_probability);
  const threshold = clamp(Number(data.threshold));
  if (![normal, pneumonia, threshold].every(Number.isFinite)) {
    throw new Error("模型返回了无效的分数。请检查服务状态。");
  }
  const isPneumonia = data.predicted_label === "PNEUMONIA";
  const distance = Math.abs(pneumonia - threshold);
  const distanceState = outputDistanceState(pneumonia, threshold);

  serviceThreshold = threshold;
  resultTitle.textContent = isPneumonia ? "PNEUMONIA标签倾向" : "NORMAL标签倾向";
  resultSummary.textContent = `PNEUMONIA标签分数为 ${formatPercent(pneumonia)}，${
    isPneumonia ? "达到" : "未达到"
  }当前判别阈值。`;
  riskPill.textContent = distanceState.label;
  statusValue.textContent = "推理完成";
  normalValue.textContent = formatPercent(normal);
  pneumoniaValue.textContent = formatPercent(pneumonia);
  normalBar.style.width = `${clamp(normal) * 100}%`;
  pneumoniaBar.style.width = `${clamp(pneumonia) * 100}%`;
  distanceValue.textContent = `${(distance * 100).toFixed(1)} 个百分点`;
  confidenceState.textContent = distanceState.label;
  scoreMarker.style.left = `${clamp(pneumonia) * 100}%`;
  scoreTag.textContent = formatPercent(pneumonia);
  scoreMarker.classList.add("visible");
  setPanelState(distanceState.state);
  updateThresholdDisplay();
}

function showError(message, title = "无法完成推理") {
  resultTitle.textContent = title;
  resultSummary.textContent = message;
  riskPill.textContent = "需要检查";
  statusValue.textContent = "出错";
  confidenceState.textContent = "不可用";
  scoreMarker.classList.remove("visible");
  setPanelState("error-state");
}

async function runPrediction() {
  if (!selectedFile || predictButton.disabled) return;
  predictButton.disabled = true;
  statusValue.textContent = "推理中";
  riskPill.textContent = "模型计算中";
  resultTitle.textContent = "正在处理图像";
  resultSummary.textContent = "正在完成预处理、模型前向计算和阈值判定。";
  confidenceState.textContent = "计算中";
  scoreMarker.classList.remove("visible");
  setPanelState();

  const formData = new FormData();
  formData.append("image", selectedFile);
  try {
    const response = await fetch("predict", { method: "POST", body: formData });
    let data;
    try {
      data = await response.json();
    } catch (_error) {
      throw new Error("模型服务返回了无法解析的响应。");
    }
    if (!response.ok || !data.ok) throw new Error(data.error || "推理失败");
    renderResult(data);
  } catch (error) {
    showError(error.message || "模型服务暂时不可用，请稍后重试。");
  } finally {
    predictButton.disabled = false;
  }
}

imageInput.addEventListener("change", () => {
  setSelectedFile(imageInput.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  setSelectedFile(event.dataTransfer.files[0]);
});

sampleButtons.forEach((button) => {
  button.addEventListener("click", () => loadSample(button.dataset.sample));
});

clearButton.addEventListener("click", () => {
  selectedFile = null;
  imageInput.value = "";
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  previewImage.src = "sample-xray/normal";
  fileName.textContent = "等待选择图像";
  fileMeta.textContent = "JPG / PNG / WebP";
  predictButton.disabled = true;
  setActiveSample();
  resetResult();
});

predictButton.addEventListener("click", runPrediction);

Promise.all([loadHealth(), loadSample("normal", false)]);
