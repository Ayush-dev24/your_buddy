const API_URL =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://your-buddy-1.onrender.com";

const API_KEY_STORAGE_KEY = "api_key";
const SESSION_ID_STORAGE_KEY = "client_session_id";
const DEFAULT_API_KEY = "__BACKEND_FALLBACK__";
const AUTH_ERROR_CODE = "auth_error";
const MASKED_KEY_TEXT = "****-****-****";

// Elements
const dropArea = document.getElementById("dropArea");
const fileUpload = document.getElementById("fileUpload");
const uploadStatus = document.getElementById("uploadStatus");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const selectedFilesList = document.getElementById("selectedFilesList");
const uploadFilesBtn = document.getElementById("uploadFilesBtn");
const toolsSection = document.getElementById("toolsSection");
const toolBtns = document.querySelectorAll(".tool-btn");
const currentModeBadge = document.getElementById("currentModeBadge");
const modeDescription = document.getElementById("modeDescription");
const statusIndicator = document.getElementById("statusIndicator");

const chatHistory = document.getElementById("chatHistory");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

const apiKeyModal = document.getElementById("apiKeyModal");
const apiKeyInput = document.getElementById("apiKeyInput");
const apiKeyContinueBtn = document.getElementById("apiKeyContinueBtn");
const apiKeyStatus = document.getElementById("apiKeyStatus");
const toastContainer = document.getElementById("toastContainer");
const toolsLockOverlay = document.getElementById("toolsLockOverlay");
const documentsList = document.getElementById("documentsList");
const documentsEmptyState = document.getElementById("documentsEmptyState");
const selectAllDocsBtn = document.getElementById("selectAllDocsBtn");
const deselectAllDocsBtn = document.getElementById("deselectAllDocsBtn");
const activeDocsSummary = document.getElementById("activeDocsSummary");

const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const addMoreBtn = document.getElementById("addMoreBtn");

let currentMode = "qa";
let isProcessing = false;
let fileUploaded = false;
let selectedFiles = [];
let processedDocuments = [];
let activeDocumentIds = new Set();
let aiResponsesDisabled = false;
let apiGateResolved = false;

// Mode Descriptions
const modeDescriptions = {
  qa: "Ask any question about your documents.",
  quiz: "Generate a 5-question quiz for self-assessment.",
  simplify: "Get simple explanations of complex topics.",
  agent: "Handle multi-step analytical tasks.",
};

const modeBadges = {
  qa: "Q&A Mode",
  quiz: "Quiz Mode",
  simplify: "Simplify Mode",
  agent: "Agent Mode",
};

// --- Global Safety Handlers ---
window.addEventListener("unhandledrejection", (event) => {
  event.preventDefault();
  showToast("Unexpected error occurred. Please try again.", "error");
});

window.addEventListener("error", () => {
  showToast("Something went wrong. Please retry.", "error");
});

// --- Initial Setup ---
initializeApiKeyGate();
setupEventListeners();
updateStatusIndicator();
refreshDocuments();

// --- Setup Event Listeners ---
function setupEventListeners() {
  if (sidebarToggle) sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("collapsed"));
  if (fileUpload) fileUpload.addEventListener("change", handleFileSelection);
  if (uploadFilesBtn) uploadFilesBtn.addEventListener("click", uploadSelectedFiles);
  if (addMoreBtn) addMoreBtn.addEventListener("click", showUploadArea);
  if (selectAllDocsBtn) selectAllDocsBtn.addEventListener("click", () => setAllDocumentsActive(true));
  if (deselectAllDocsBtn) deselectAllDocsBtn.addEventListener("click", () => setAllDocumentsActive(false));

  // Drag & Drop
  if (dropArea) {
    dropArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropArea.classList.add("dragover");
    });

    dropArea.addEventListener("dragleave", () => {
      dropArea.classList.remove("dragover");
    });

    dropArea.addEventListener("drop", (e) => {
      e.preventDefault();
      dropArea.classList.remove("dragover");

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        selectedFiles = Array.from(e.dataTransfer.files);
        if (fileUpload) fileUpload.value = "";
        renderSelectedFiles();
      }
    });
  }

  // Tool Selection
  if (toolBtns.length) {
    toolBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        if (!fileUploaded) return;

        toolBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");

        currentMode = btn.dataset.mode || "qa";
        if (currentModeBadge) currentModeBadge.textContent = modeBadges[currentMode] || "Q&A Mode";
        if (modeDescription) modeDescription.textContent = modeDescriptions[currentMode] || "";

        if (currentMode === "qa") userInput.placeholder = "Ask a question...";
        if (currentMode === "quiz") userInput.placeholder = "e.g., 'Chapter 1' or 'Photosynthesis'";
        if (currentMode === "simplify") userInput.placeholder = "What should I simplify?";
        if (currentMode === "agent") userInput.placeholder = "Describe the task...";

        userInput.focus();
      });
    });
  }

  // Chat Submit
  if (chatForm && userInput && sendBtn) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!apiGateResolved) {
        showToast("Set API key first to continue.", "warning");
        return;
      }
      if (!userInput.value.trim() || isProcessing || !fileUploaded) return;
      if (!activeDocumentIds.size) {
        showToast("Please select at least one document to generate an answer.", "warning");
        return;
      }
      await processUserQuery(userInput.value.trim());
    });

    userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit"));
      }
    });

    userInput.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = this.scrollHeight + "px";
      if (this.value.trim() === "") {
        this.style.height = "48px";
      }
    });
  } else {
    showToast("Chat elements missing from page.", "error");
  }
}

// --- API Key Gate ---
function initializeApiKeyGate() {
  if (!apiKeyModal || !apiKeyInput || !apiKeyContinueBtn) {
    apiGateResolved = true;
    updateApiKeyStatus(getStoredApiKey());
    return;
  }

  const savedKey = getStoredApiKey();
  if (savedKey && savedKey !== DEFAULT_API_KEY) {
    apiKeyInput.value = savedKey;
  }

  updateContinueButtonState();
  updateApiKeyStatus(savedKey);

  apiKeyInput.addEventListener("input", updateContinueButtonState);
  apiKeyInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !apiKeyContinueBtn.disabled) {
      e.preventDefault();
      handleApiKeyContinue();
    }
  });

  apiKeyContinueBtn.addEventListener("click", handleApiKeyContinue);

  if (apiKeyStatus) {
    apiKeyStatus.addEventListener("click", () => {
      apiGateResolved = false;
      apiKeyModal.classList.remove("hidden");
      const saved = getStoredApiKey();
      apiKeyInput.value = saved || "";
      updateContinueButtonState();
      apiKeyInput.focus();
    });
  }

  apiKeyInput.focus();
}

function updateContinueButtonState() {
  if (!apiKeyContinueBtn || !apiKeyInput) return;
  const hasInput = apiKeyInput.value.trim().length > 0;
  apiKeyContinueBtn.disabled = !hasInput;
}

function handleApiKeyContinue() {
  const enteredKey = apiKeyInput ? apiKeyInput.value.trim() : "";

  if (!enteredKey) {
    alert("Gemini API key is required.");
    showToast("Enter a valid Gemini API key.", "warning");
    return;
  }

  storeApiKey(enteredKey);
  apiGateResolved = true;
  aiResponsesDisabled = false;
  updateStatusIndicator();
  updateApiKeyStatus(enteredKey);

  if (apiKeyModal) apiKeyModal.classList.add("hidden");
  showToast("Gemini API key configured.", "success");
}

function updateApiKeyStatus(keyValue) {
  if (!apiKeyStatus) return;
  if (!keyValue) {
    apiKeyStatus.classList.add("hidden");
    apiKeyStatus.textContent = "Key: Not set";
    return;
  }
  apiKeyStatus.classList.remove("hidden");
  apiKeyStatus.textContent = `Key: ${MASKED_KEY_TEXT}`;
}

function getStoredApiKey() {
  try {
    return (localStorage.getItem(API_KEY_STORAGE_KEY) || "").trim();
  } catch {
    return "";
  }
}

function storeApiKey(key) {
  try {
    localStorage.setItem(API_KEY_STORAGE_KEY, key);
  } catch {
    showToast("Could not save API key in browser storage.", "warning");
  }
}

function resolvePrimaryApiKey() {
  const userKey = getStoredApiKey();
  return userKey || DEFAULT_API_KEY;
}

function normalizeApiKeyForHeader(key) {
  const normalized = String(key || "").trim();
  if (!normalized || normalized === DEFAULT_API_KEY) return "";
  return normalized;
}

function getOrCreateSessionId() {
  try {
    let existing = (localStorage.getItem(SESSION_ID_STORAGE_KEY) || "").trim();
    if (existing) return existing;

    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      existing = window.crypto.randomUUID();
    } else {
      existing = `session-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
    }
    localStorage.setItem(SESSION_ID_STORAGE_KEY, existing);
    return existing;
  } catch {
    return `session-${Date.now()}`;
  }
}

function buildRequestHeaders(baseHeaders, apiKey) {
  const headers = new Headers(baseHeaders || {});
  const normalizedKey = normalizeApiKeyForHeader(apiKey);
  if (normalizedKey) {
    headers.set("Authorization", `Bearer ${normalizedKey}`);
  }
  headers.set("x-user-id", getOrCreateSessionId());
  return headers;
}

function looksLikeAuthError(response, body, message) {
  if (response && response.status === 401) return true;

  if (body && typeof body === "object") {
    const code = String(body.code || "").toLowerCase();
    if (code === AUTH_ERROR_CODE) return true;
  }

  const text = String(message || "").toLowerCase();
  return /(auth_error|unauthorized|forbidden|invalid api key|api key not valid|authentication|permission denied)/i.test(
    text
  );
}

async function sendApiRequest(path, init, apiKey) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: buildRequestHeaders(init.headers, apiKey),
    });
  } catch {
    return {
      ok: false,
      authError: false,
      message: "Network issue. Please check connection and try again.",
      body: null,
      response: null,
    };
  }

  let parsedBody = null;
  let rawText = "";
  try {
    parsedBody = await response.json();
  } catch {
    try {
      rawText = await response.text();
    } catch {
      rawText = "";
    }
  }

  const message =
    (parsedBody && (parsedBody.message || parsedBody.detail)) ||
    rawText ||
    `Request failed (${response.status})`;
  const authError = looksLikeAuthError(response, parsedBody, message);
  const ok = response.ok && (!parsedBody || parsedBody.status !== "error");

  return {
    ok,
    authError,
    message,
    body: parsedBody,
    rawText,
    response,
  };
}

async function sendApiRequestWithFallback(path, init, options = {}) {
  const { aiRequest = false } = options;
  const primaryKey = resolvePrimaryApiKey();
  const first = await sendApiRequest(path, init, primaryKey);
  if (!first.authError) return first;

  alert("API key failed. Switching to fallback mode.");
  showToast("API key failed. Trying default fallback key.", "warning");

  const second = await sendApiRequest(path, init, DEFAULT_API_KEY);
  if (second.authError && aiRequest) {
    enableAiFallbackMode();
  }
  return second;
}

function enableAiFallbackMode() {
  aiResponsesDisabled = true;
  updateStatusIndicator();
  showToast("AI responses are disabled. Update API key to re-enable.", "error");
}

function updateStatusIndicator() {
  if (!statusIndicator) return;
  if (aiResponsesDisabled) {
    statusIndicator.textContent = "Fallback mode";
    statusIndicator.style.color = "#f59e0b";
    return;
  }
  statusIndicator.textContent = "Ready";
  statusIndicator.style.color = "";
}

// --- Upload / Query Functions ---
function handleFileSelection() {
  selectedFiles = Array.from((fileUpload && fileUpload.files) || []);
  renderSelectedFiles();
}

function renderSelectedFiles() {
  if (!selectedFiles.length) {
    if (selectedFilesList) selectedFilesList.classList.add("hidden");
    if (uploadFilesBtn) uploadFilesBtn.classList.add("hidden");
    return;
  }

  if (selectedFilesList) selectedFilesList.classList.remove("hidden");
  if (uploadFilesBtn) {
    uploadFilesBtn.classList.remove("hidden");
    uploadFilesBtn.disabled = false;
  }

  let filesHtml = `
    <div class="files-header">
      <i class="fas fa-check-circle"></i>
      <span id="filesCount">${selectedFiles.length} file${selectedFiles.length !== 1 ? "s" : ""} selected</span>
    </div>
  `;

  selectedFiles.forEach((file, index) => {
    filesHtml += `
      <div class="selected-file-item">
        <span title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
        <button type="button" class="remove-file-btn" onclick="removeSelectedFile(${index})">Remove</button>
      </div>
    `;
  });

  if (selectedFilesList) selectedFilesList.innerHTML = filesHtml;
}

window.removeSelectedFile = function (index) {
  selectedFiles.splice(index, 1);
  if (!selectedFiles.length && fileUpload) {
    fileUpload.value = "";
  }
  renderSelectedFiles();
};

function showUploadArea() {
  if (fileInfo) fileInfo.classList.add("hidden");
  if (dropArea) dropArea.classList.remove("hidden");
  selectedFiles = [];
  if (fileUpload) fileUpload.value = "";
  renderSelectedFiles();
}

async function uploadSelectedFiles() {
  if (!apiGateResolved) {
    showToast("Set API key first to continue.", "warning");
    return;
  }

  if (!selectedFiles.length) {
    alert("Please select at least one file to upload.");
    return;
  }

  if (dropArea) dropArea.classList.add("hidden");
  if (uploadStatus) uploadStatus.classList.remove("hidden");

  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));

  try {
    const request = await sendApiRequestWithFallback(
      "/upload",
      {
        method: "POST",
        body: formData,
      },
      { aiRequest: false }
    );

    if (!request.ok) {
      throw new Error(request.message || "Upload failed");
    }

    const result = request.body || {};
    if (uploadStatus) uploadStatus.classList.add("hidden");

    selectedFiles = [];
    if (fileUpload) fileUpload.value = "";
    renderSelectedFiles();

    if (fileInfo) fileInfo.classList.remove("hidden");
    if (fileName) {
      fileName.textContent = `Uploaded ${result.files_received || 1} file(s), created ${result.chunks_created || 0} chunks`;
    }
    if (toolsLockOverlay) toolsLockOverlay.classList.add("hidden");

    if (userInput) userInput.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    fileUploaded = true;
    if (Array.isArray(result.documents)) {
      result.documents.forEach((doc) => {
        if (doc && doc.id) activeDocumentIds.add(String(doc.id));
      });
    }
    await refreshDocuments();

    appendBotMessage(
      "Files processed successfully. The knowledge base is ready. What would you like to do?"
    );
  } catch (error) {
    alert(`Error uploading files: ${error.message || "Upload failed"}`);
    if (uploadStatus) uploadStatus.classList.add("hidden");
    if (dropArea) dropArea.classList.remove("hidden");
  }
}

async function processUserQuery(query) {
  appendUserMessage(query);
  userInput.value = "";
  userInput.style.height = "48px";

  if (aiResponsesDisabled) {
    appendBotMessage(
      "AI responses are currently disabled due to authentication failures. Update your API key to continue."
    );
    userInput.focus();
    return;
  }

  isProcessing = true;
  sendBtn.disabled = true;
  const loadingId = appendLoadingMessage();

  try {
    const request = await sendApiRequestWithFallback(
      "/query",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          mode: currentMode,
          selected_document_ids: Array.from(activeDocumentIds),
        }),
      },
      { aiRequest: true }
    );

    removeLoadingMessage(loadingId);

    if (request.authError && aiResponsesDisabled) {
      appendBotMessage(
        "API key authentication failed. Fallback mode is active and AI responses are disabled."
      );
      return;
    }

    if (!request.ok) {
      throw new Error(request.message || "Failed to process query");
    }

    const result = request.body || {};
    if (result.status === "error") {
      appendBotMessage(result.message || "Something went wrong.");
      return;
    }

    const payload = result.response;

    if (currentMode === "quiz") {
      const quizData = parseQuizResponse(payload);
      if (Array.isArray(quizData)) {
        renderInteractiveQuiz(quizData);
      } else {
        appendBotMessage("Server returned invalid quiz format.\n\n" + String(payload || ""));
      }
    } else {
      appendBotMessage(String(payload || "No response received."));
    }
  } catch (error) {
    removeLoadingMessage(loadingId);
    appendBotMessage("Server error:\n\n" + (error.message || "Something went wrong."));
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

function parseQuizResponse(payload) {
  if (!payload) return null;

  if (Array.isArray(payload)) return payload;
  if (typeof payload === "object") {
    if (payload.status === "error") return payload;
    if (Array.isArray(payload.data)) return payload.data;
  }

  if (typeof payload !== "string") return null;

  let jsonStr = payload.trim();

  if (jsonStr.startsWith("```")) {
    jsonStr = jsonStr.replace(/```json|```/g, "").trim();
  }

  try {
    return JSON.parse(jsonStr);
  } catch {
    return null;
  }
}

// --- Quiz UI Logic ---
function renderInteractiveQuiz(quizData) {
  const div = document.createElement("div");
  div.className = "message bot-message";

  let quizHtml = `
    <div class="avatar"><i class="fas fa-robot"></i></div>
    <div class="content quiz-container">
      <h3 style="margin-top: 0; margin-bottom: 1.5rem;"><i class="fas fa-list-check"></i> Interactive Quiz</h3>
      <div id="quizQuestions">
  `;

  quizData.forEach((q, qIndex) => {
    quizHtml += `
      <div class="quiz-question" data-answer="${q.answer_index}">
        <p><strong>Q${qIndex + 1}: ${escapeHtml(q.question)}</strong></p>
        <div class="quiz-options">
    `;
    q.options.forEach((opt, oIndex) => {
      quizHtml += `
        <button class="quiz-option" onclick="selectQuizOption(this, ${qIndex}, ${oIndex})">
          <span class="opt-label">${String.fromCharCode(65 + oIndex)}</span> ${escapeHtml(opt)}
        </button>
      `;
    });
    quizHtml += `</div></div>`;
  });

  quizHtml += `
      </div>
      <button class="quiz-submit-btn" onclick="submitQuiz(this, ${quizData.length})">Submit Answers</button>
      <div class="quiz-result hidden"></div>
    </div>
  `;

  div.innerHTML = quizHtml;
  chatHistory.appendChild(div);
  scrollToBottom();
}

window.selectQuizOption = function (btn, qIndex, oIndex) {
  const optionsContainer = btn.parentElement;
  Array.from(optionsContainer.children).forEach((child) => child.classList.remove("selected"));
  btn.classList.add("selected");
  optionsContainer.dataset.answered = oIndex;
};

window.submitQuiz = function (submitBtn, totalQuestions) {
  const container = submitBtn.parentElement;
  const questions = container.querySelectorAll(".quiz-question");
  let score = 0;
  let allAnswered = true;

  questions.forEach((q) => {
    const optionsContainer = q.querySelector(".quiz-options");
    if (optionsContainer.dataset.answered === undefined) {
      allAnswered = false;
    }
  });

  if (!allAnswered) {
    alert("Please select an answer for all questions before submitting.");
    return;
  }

  questions.forEach((q) => {
    const correctIndex = parseInt(q.dataset.answer, 10);
    const optionsContainer = q.querySelector(".quiz-options");
    const options = optionsContainer.querySelectorAll(".quiz-option");
    const answeredIndex = parseInt(optionsContainer.dataset.answered, 10);

    options.forEach((opt, idx) => {
      opt.disabled = true;
      if (idx === correctIndex) {
        opt.classList.add("correct");
      } else if (idx === answeredIndex && answeredIndex !== correctIndex) {
        opt.classList.add("wrong");
      }
    });

    if (answeredIndex === correctIndex) {
      score++;
    }
  });

  submitBtn.classList.add("hidden");
  const resultDiv = container.querySelector(".quiz-result");
  resultDiv.classList.remove("hidden");
  const scoreColor = score === totalQuestions ? "#10b981" : "#a855f7";
  resultDiv.innerHTML = `<h4 style="color: ${scoreColor}; font-size: 1.2rem; margin: 0;">You scored ${score} out of ${totalQuestions}.</h4>`;
};

// --- UI Helpers ---
function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "message user-message";
  div.innerHTML = `
    <div class="avatar"><i class="fas fa-user"></i></div>
    <div class="content"><p>${escapeHtml(String(text))}</p></div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
}

function appendBotMessage(text) {
  const div = document.createElement("div");
  div.className = "message bot-message";

  let formattedText = `<p>${escapeHtml(String(text))}</p>`;
  if (typeof marked !== "undefined") {
    try {
      if (typeof marked.parse === "function") {
        formattedText = marked.parse(String(text));
      } else if (typeof marked === "function") {
        formattedText = marked(String(text));
      }
    } catch {
      formattedText = `<p>${escapeHtml(String(text))}</p>`;
    }
  }

  div.innerHTML = `
    <div class="avatar"><i class="fas fa-robot"></i></div>
    <div class="content">${formattedText}</div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
}

function appendLoadingMessage() {
  const id = "loading-" + Date.now();
  const div = document.createElement("div");
  div.className = "message bot-message";
  div.id = id;
  div.innerHTML = `
    <div class="avatar"><i class="fas fa-robot"></i></div>
    <div class="content loader">
      <div class="dot"></div>
      <div class="dot"></div>
      <div class="dot"></div>
    </div>
  `;
  chatHistory.appendChild(div);
  scrollToBottom();
  return id;
}

function removeLoadingMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(unsafe) {
  return String(unsafe)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function showToast(message, type = "info") {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `app-toast ${type}`;
  toast.textContent = String(message || "");
  toastContainer.appendChild(toast);

  window.setTimeout(() => {
    toast.remove();
  }, 3200);
}

async function refreshDocuments() {
  const request = await sendApiRequestWithFallback(
    "/documents",
    { method: "GET" },
    { aiRequest: false }
  );

  if (!request.ok) {
    processedDocuments = [];
    renderDocumentsList();
    return;
  }

  processedDocuments = Array.isArray(request.body?.documents) ? request.body.documents : [];
  const available = new Set(processedDocuments.map((d) => String(d.id)));
  activeDocumentIds.forEach((id) => {
    if (!available.has(String(id))) activeDocumentIds.delete(String(id));
  });

  fileUploaded = processedDocuments.length > 0;
  if (toolsLockOverlay) toolsLockOverlay.classList.toggle("hidden", fileUploaded);
  if (userInput) userInput.disabled = !fileUploaded;
  if (sendBtn) sendBtn.disabled = !fileUploaded;
  if (fileInfo && !fileUploaded) fileInfo.classList.add("hidden");
  renderDocumentsList();
}

function setAllDocumentsActive(active) {
  if (!processedDocuments.length) return;
  if (active) {
    processedDocuments.forEach((doc) => activeDocumentIds.add(String(doc.id)));
  } else {
    activeDocumentIds.clear();
  }
  renderDocumentsList();
}

async function removeDocument(documentId) {
  const request = await sendApiRequestWithFallback(
    `/documents/${encodeURIComponent(String(documentId))}`,
    { method: "DELETE" },
    { aiRequest: false }
  );

  if (!request.ok) {
    showToast(request.message || "Failed to remove document.", "error");
    return;
  }

  activeDocumentIds.delete(String(documentId));
  await refreshDocuments();
  showToast("Document removed.", "success");
}

function renderDocumentsList() {
  const hasDocuments = processedDocuments.length > 0;
  if (documentsEmptyState) {
    documentsEmptyState.classList.toggle("hidden", hasDocuments);
  }
  if (!documentsList) return;

  if (!hasDocuments) {
    documentsList.innerHTML = "";
    updateActiveDocsSummary();
    return;
  }

  documentsList.innerHTML = processedDocuments
    .map((doc) => {
      const id = String(doc.id || "");
      const name = escapeHtml(doc.name || "upload");
      const isActive = activeDocumentIds.has(id);
      const chunkCount = Number(doc.chunk_count || 0);
      return `
        <div class="doc-item ${isActive ? "active" : ""}" data-doc-id="${escapeHtml(id)}">
          <label class="doc-checkbox-wrap">
            <input type="checkbox" class="doc-checkbox" data-doc-id="${escapeHtml(id)}" ${isActive ? "checked" : ""} />
            <span class="doc-name" title="${name}">${name}</span>
          </label>
          <div class="doc-meta-row">
            <span class="doc-tag">${chunkCount} chunks</span>
            <span class="doc-tag ${isActive ? "active-tag" : ""}">${isActive ? "Active" : "Inactive"}</span>
            <button type="button" class="doc-delete-btn" data-doc-delete-id="${escapeHtml(id)}">Delete</button>
          </div>
        </div>
      `;
    })
    .join("");

  documentsList.querySelectorAll(".doc-checkbox").forEach((checkbox) => {
    checkbox.addEventListener("change", (e) => {
      const id = String(e.target.getAttribute("data-doc-id") || "");
      if (!id) return;
      if (e.target.checked) activeDocumentIds.add(id);
      else activeDocumentIds.delete(id);
      renderDocumentsList();
    });
  });

  documentsList.querySelectorAll(".doc-delete-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const id = String(e.target.getAttribute("data-doc-delete-id") || "");
      if (!id) return;
      await removeDocument(id);
    });
  });

  updateActiveDocsSummary();
}

function updateActiveDocsSummary() {
  if (!activeDocsSummary) return;
  const activeCount = activeDocumentIds.size;
  if (!activeCount) {
    activeDocsSummary.textContent = "Active Documents: None selected";
    return;
  }

  const activeNames = processedDocuments
    .filter((doc) => activeDocumentIds.has(String(doc.id)))
    .map((doc) => String(doc.name || "upload"));
  activeDocsSummary.textContent = `Active Documents (${activeCount}): ${activeNames.join(", ")}`;
}

function toggleSection(contentId, header) {
  const content = document.getElementById(contentId);
  if (!content) return;

  const isCollapsed = content.classList.toggle("collapsed");
  if (header) {
    header.classList.toggle("collapsed", isCollapsed);
  }
}

// Ensure smooth height transitions by recalculating if needed
window.addEventListener("resize", () => {
  document.querySelectorAll(".collapsible-content:not(.collapsed)").forEach((el) => {
    el.style.maxHeight = "1000px";
  });
});
