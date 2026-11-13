const videoInput = document.getElementById("videoInput");
const uploadForm = document.getElementById("uploadForm");
const fileName = document.getElementById("fileName");
const analyzeButton = document.getElementById("analyzeButton");
const analysisStatus = document.getElementById("analysisStatus");
const resultsSection = document.getElementById("results");

// Change this if your FastAPI backend uses a different address.
const API_URL = "http://127.0.0.1:8000/analyze";

// Display the selected video filename.
videoInput.addEventListener("change", () => {
    const file = videoInput.files[0];

    fileName.textContent = file
        ? `Selected video: ${file.name}`
        : "No video selected";

    analysisStatus.textContent = "";
    analysisStatus.classList.remove("error");
});

// Submit the selected video to the FastAPI backend.
uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = videoInput.files[0];

    if (!file) {
        showStatus("Please select a video first.", true);
        return;
    }

    const allowedExtensions = /\.(mp4|avi|mov|mkv|webm)$/i;

    if (!allowedExtensions.test(file.name)) {
        showStatus(
            "Please select a supported video file: MP4, AVI, MOV, MKV, or WebM.",
            true
        );
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing Video...";
    resultsSection.hidden = true;

    showStatus(
        "Uploading video and running DeepTrace analysis. Please wait...",
        false
    );

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(
                data.detail ||
                data.message ||
                `Server returned error ${response.status}.`
            );
        }

        if (data.status !== "success") {
            throw new Error(
                data.message || "The analysis did not complete successfully."
            );
        }

        displayResults(data);

        resultsSection.hidden = false;
        showStatus("Analysis completed successfully!", false);

        resultsSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {
        console.error("DeepTrace API error:", error);

        showStatus(
            "Analysis failed. Make sure the FastAPI backend is running. " +
            "Details: " + error.message,
            true
        );

    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Video →";
    }
});

// Display results returned by the backend.
function displayResults(data) {
    const prediction = document.getElementById("prediction");
    const confidence = document.getElementById("confidence");

    prediction.textContent = data.prediction || "Unavailable";

    prediction.style.color =
        String(data.prediction).toUpperCase() === "REAL"
            ? "#49e6a5"
            : String(data.prediction).toUpperCase() === "DEEPFAKE"
                ? "#ff7185"
                : "#f4f7ff";

    confidence.textContent =
        data.confidence !== undefined
            ? `${Number(data.confidence).toFixed(2)}%`
            : "Unavailable";

    document.getElementById("framesAnalyzed").textContent =
        data.frames_analyzed ?? "Unavailable";

    document.getElementById("temporalAssessment").textContent =
        data.temporal_assessment ?? "Unavailable";

    document.getElementById("suspiciousFrames").textContent =
        data.suspicious_frames ?? "Unavailable";

    document.getElementById("evidenceScore").textContent =
        data.combined_evidence_score !== undefined
            ? Number(data.combined_evidence_score).toFixed(4)
            : "Unavailable";
}

// Show status messages to the user.
function showStatus(message, isError) {
    analysisStatus.textContent = message;
    analysisStatus.classList.toggle("error", isError);
}