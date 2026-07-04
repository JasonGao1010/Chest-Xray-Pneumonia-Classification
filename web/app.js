const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const previewImage = document.querySelector("#previewImage");
const predictButton = document.querySelector("#predictButton");
const clearButton = document.querySelector("#clearButton");
const fileName = document.querySelector("#fileName");
const fileMeta = document.querySelector("#fileMeta");
const resultPanel = document.querySelector(".result-panel");
const resultTitle = document.querySelector("#resultTitle");
const resultSummary = document.querySelector("#resultSummary");
const riskPill = document.querySelector("#riskPill");
const normalValue = document.querySelector("#normalValue");
const pneumoniaValue = document.querySelector("#pneumoniaValue");
const normalBar = document.querySelector("#normalBar");
const pneumoniaBar = document.querySelector("#pneumoniaBar");
const thresholdValue = document.querySelector("#thresholdValue");
const confidenceValue = document.querySelector("#confidenceValue");
const statusValue = document.querySelector("#statusValue");

let selectedFile = null;
let objectUrl = null;

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function clampPercent(value) {
  return Math.max(0, Math.min(100, value * 100));
}

function setBars(normal, pneumonia) {
  normalValue.textContent = formatPercent(normal);
  pneumoniaValue.textContent = formatPercent(pneumonia);
  normalBar.style.width = typeof normal === "number" ? `${clampPercent(normal)}%` : "0%";
  pneumoniaBar.style.width = typeof pneumonia === "number" ? `${clampPercent(pneumonia)}%` : "0%";
}

function setPanelState(state) {
  resultPanel.classList.remove("normal-state", "pneumonia-state", "error-state");
  if (state) {
    resultPanel.classList.add(state);
  }
}

function resetResult() {
  resultTitle.textContent = "尚未分类";
  resultSummary.textContent = "选择胸部 X 光图片后，模型会返回正常类和肺炎类的概率。";
  riskPill.textContent = "等待图片";
  thresholdValue.textContent = "0.50";
  confidenceValue.textContent = "--";
  statusValue.textContent = "未运行";
  setBars(null, null);
  setPanelState("");
}

function setSelectedFile(file) {
  if (!file) {
    return;
  }
  selectedFile = file;
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }
  objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  fileName.textContent = file.name;
  fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  predictButton.disabled = false;
  resetResult();
}

imageInput.addEventListener("change", () => {
  setSelectedFile(imageInput.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  setSelectedFile(event.dataTransfer.files[0]);
});

clearButton.addEventListener("click", () => {
  selectedFile = null;
  imageInput.value = "";
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
  previewImage.src = "sample-xray";
  fileName.textContent = "示例胸片";
  fileMeta.textContent = "等待选择";
  predictButton.disabled = true;
  resetResult();
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  predictButton.disabled = true;
  statusValue.textContent = "推理中";
  riskPill.textContent = "计算中";
  resultTitle.textContent = "正在分类";
  resultSummary.textContent = "模型正在读取图片并计算类别概率。";
  confidenceValue.textContent = "--";
  setPanelState("");

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "分类失败");
    }

    const normal = data.normal_probability;
    const pneumonia = data.pneumonia_probability;
    const confidence = data.confidence;
    const isPneumonia = data.predicted_label === "PNEUMONIA";
    resultTitle.textContent = data.predicted_label_cn;
    resultSummary.textContent = isPneumonia
      ? `模型更倾向于肺炎类，肺炎概率为 ${formatPercent(pneumonia)}。`
      : `模型更倾向于正常类，肺炎概率为 ${formatPercent(pneumonia)}。`;
    riskPill.textContent = isPneumonia ? "肺炎倾向" : "正常倾向";
    thresholdValue.textContent = Number(data.threshold).toFixed(2);
    confidenceValue.textContent = formatPercent(confidence);
    statusValue.textContent = "已完成";
    setBars(normal, pneumonia);
    setPanelState(isPneumonia ? "pneumonia-state" : "normal-state");
  } catch (error) {
    resultTitle.textContent = "无法分类";
    resultSummary.textContent = error.message;
    riskPill.textContent = "需要重试";
    statusValue.textContent = "出错";
    confidenceValue.textContent = "--";
    setPanelState("error-state");
  } finally {
    predictButton.disabled = false;
  }
});
