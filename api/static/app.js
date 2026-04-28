const manualForm = document.getElementById("manual-form");
const audioForm = document.getElementById("audio-form");

const statusCard = document.getElementById("status-card");
const statusMessage = document.getElementById("status-message");

const resultCard = document.getElementById("result-card");
const severityBadge = document.getElementById("severity-badge");
const severityClass = document.getElementById("severity-class");
const confidenceValue = document.getElementById("confidence-value");
const inputSource = document.getElementById("input-source");
const probabilityBars = document.getElementById("probability-bars");

function showStatus(message, type = "info") {
    statusMessage.textContent = message;
    statusCard.classList.remove("hidden", "success", "error");
    if (type === "success") statusCard.classList.add("success");
    if (type === "error") statusCard.classList.add("error");
}

function hideStatus() {
    statusCard.classList.add("hidden");
    statusCard.classList.remove("success", "error");
}

function normalizeSeverityClass(label = "") {
    return label.toLowerCase().trim();
}

function renderProbabilities(probabilities = {}) {
    probabilityBars.innerHTML = "";
    Object.entries(probabilities).forEach(([label, value]) => {
        const percent = Math.round((Number(value) || 0) * 100);
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <span>${label}</span>
            <div class="prob-track"><div class="prob-fill" style="width: ${Math.min(100, Math.max(0, percent))}%"></div></div>
            <strong>${percent}%</strong>
        `;
        probabilityBars.appendChild(row);
    });
}

function renderResult(result) {
    resultCard.classList.remove("hidden");
    const label = result.severity_label || "Unknown";
    const score = Math.round((Number(result.confidence) || 0) * 100);
    severityBadge.textContent = label;
    severityBadge.className = `severity-badge ${normalizeSeverityClass(label)}`;
    severityClass.textContent = result.severity_class ?? "-";
    confidenceValue.textContent = `${score}%`;
    inputSource.textContent = result.input_source || "-";
    renderProbabilities(result.probabilities || {});
}

function collectManualPayload() {
    const data = Object.fromEntries(new FormData(manualForm).entries());
    const payload = {};

    for (const [key, value] of Object.entries(data)) {
        const numeric = Number(value);
        if (Number.isNaN(numeric)) {
            throw new Error(`"${key}" must be numeric.`);
        }

        if (key.startsWith("hearing_") && (numeric < 0 || numeric > 120)) {
            throw new Error(`"${key}" must be between 0 and 120 dB HL.`);
        }
        if (key === "age" && (numeric < 0 || numeric > 120)) {
            throw new Error("Age must be between 0 and 120.");
        }
        if (key === "noise_exposure" && (numeric < 0 || numeric > 60)) {
            throw new Error("Noise exposure must be between 0 and 60 years.");
        }
        if ((key === "gender" || key === "tinnitus") && ![0, 1].includes(numeric)) {
            throw new Error(`"${key}" must be 0 or 1.`);
        }
        payload[key] = numeric;
    }
    return payload;
}

manualForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideStatus();
    resultCard.classList.add("hidden");
    showStatus("Running manual prediction...");

    try {
        const payload = collectManualPayload();
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed.");
        renderResult(data);
        showStatus("Prediction complete.", "success");
    } catch (error) {
        showStatus(error.message, "error");
    }
});

audioForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideStatus();
    resultCard.classList.add("hidden");
    showStatus("Uploading audio and running prediction...");

    try {
        const fileInput = document.getElementById("audio-file");
        if (!fileInput.files || fileInput.files.length === 0) {
            throw new Error("Choose an audio file first.");
        }
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const response = await fetch("/api/predict/audio", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Audio prediction failed.");
        renderResult(data);
        showStatus("Prediction complete.", "success");
    } catch (error) {
        showStatus(error.message, "error");
    }
});
