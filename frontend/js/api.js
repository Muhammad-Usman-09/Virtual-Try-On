/**
 * Shared API utility functions
 * All pages import this file to talk to the Flask backend
 */

const API_BASE = 'http://localhost:5000/api';

/**
 * Upload an image file + form data to a backend endpoint.
 * Returns parsed JSON response.
 */
async function apiUploadImage(endpoint, imageFile, extraData = {}) {
  const formData = new FormData();
  formData.append('image', imageFile);
  for (const [key, value] of Object.entries(extraData)) {
    formData.append(key, value);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    body: formData   // Don't set Content-Type — browser sets it with boundary
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Send JSON data to a backend endpoint.
 */
async function apiPost(endpoint, data) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Fetch data from a GET endpoint.
 */
async function apiGet(endpoint) {
  const response = await fetch(`${API_BASE}${endpoint}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

/**
 * Show error message in a container element
 */
function showError(containerId, message) {
  const el = document.getElementById(containerId);
  if (el) {
    el.innerHTML = `<div class="error-msg">⚠️ ${message}</div>`;
    el.style.display = 'block';
  }
}

/**
 * Show loading spinner in result area
 */
function showLoading(containerId, message = 'AI is processing...') {
  const el = document.getElementById(containerId);
  if (el) {
    el.innerHTML = `
      <div class="loading-box">
        <div class="spinner"></div>
        <p>${message}</p>
      </div>`;
    el.style.display = 'flex';
  }
}

/**
 * Display result image
 */
function showResultImage(containerId, base64Image, label = '') {
  const el = document.getElementById(containerId);
  if (el) {
    el.innerHTML = `
      <div class="result-wrap">
        ${label ? `<p class="result-label">${label}</p>` : ''}
        <img src="${base64Image}" alt="Try-on result" class="result-img">
        <a href="${base64Image}" download="tryon-result.jpg" class="download-btn">⬇ Download Result</a>
      </div>`;
    el.style.display = 'block';
  }
}

/**
 * Preview uploaded image before sending to API
 */
function previewImage(fileInput, previewId) {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById(previewId);
    if (preview) {
      preview.src = e.target.result;
      preview.style.display = 'block';
    }
  };
  reader.readAsDataURL(file);
}

/**
 * Seed demo data on first load (call once)
 */
async function seedDemoData() {
  try {
    const result = await apiPost('/inventory/seed', {});
    console.log('Demo data:', result.message);
  } catch (err) {
    console.log('Seed skipped (may already exist):', err.message);
  }
}
