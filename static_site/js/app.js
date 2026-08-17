/* ═══════════════════════════════════════════════════════════════════
   CreditIQ — Static Web App JavaScript Engine
   Connects to Render Flask Backend API
   ═══════════════════════════════════════════════════════════════════ */

// State Management
const DEFAULT_API_BASE = "https://credit-risk-assesment-1.onrender.com";

const state = {
  apiBase: localStorage.getItem("creditiq_api_base") || DEFAULT_API_BASE,
  token: localStorage.getItem("creditiq_token") || null,
  user: JSON.parse(localStorage.getItem("creditiq_user") || "null"),
  currentView: "home",
  applications: [],
  adminApplications: [],
  selectedApp: null
};

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  initApiUrlInput();
  updateAuthUI();
  setupNavigation();
  setupForms();
  checkApiHealth();

  // Load initial view
  switchView(state.currentView);
});

/* ─── API Helper ───────────────────────────────────────────────────────────── */

function getApiUrl(endpoint) {
  const base = state.apiBase.replace(/\/$/, "");
  const path = endpoint.startsWith("/") ? endpoint : "/" + endpoint;
  return `${base}${path}`;
}

async function apiFetch(endpoint, options = {}) {
  const url = getApiUrl(endpoint);
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {})
  };

  if (state.token) {
    headers["Authorization"] = `Bearer ${state.token}`;
  }

  try {
    const response = await fetch(url, { ...options, headers });
    const data = await response.json();

    if (response.status === 401) {
      showToast("Session expired or unauthorized. Please log in.", "error");
      logout();
      return null;
    }

    if (!response.ok) {
      throw new Error(data.error || data.message || `Request failed (${response.status})`);
    }

    return data;
  } catch (err) {
    console.error("API Error:", err);
    showToast(err.message || "Network error. Please check backend connection.", "error");
    throw err;
  }
}

async function checkApiHealth() {
  const indicator = document.getElementById("api-status-indicator");
  if (!indicator) return;
  
  indicator.textContent = "Connecting...";
  indicator.className = "role-pill";
  
  try {
    const res = await apiFetch("/api/health");
    if (res && res.status === "running") {
      indicator.textContent = "API Online";
      indicator.classList.add("role-user");
    } else {
      indicator.textContent = "API Offline";
      indicator.classList.add("role-admin");
    }
  } catch {
    indicator.textContent = "API Unreachable";
    indicator.classList.add("role-admin");
  }
}

/* ─── Auth Functions ───────────────────────────────────────────────────────── */

async function loginUser(email, password) {
  try {
    const res = await apiFetch("/api/login", {
      method: "POST",
      body: JSON.stringify({ email, password })
    });

    if (res && res.success) {
      state.token = res.token;
      state.user = res.user;
      localStorage.setItem("creditiq_token", res.token);
      localStorage.setItem("creditiq_user", JSON.stringify(res.user));

      updateAuthUI();
      closeModal("auth-modal");
      showToast(`Welcome back, ${res.user.name}!`, "success");

      if (res.user.role === "admin") {
        switchView("admin");
      } else {
        switchView("apply");
      }
    }
  } catch (err) {
    // Handled in apiFetch
  }
}

async function registerUser(name, email, password) {
  try {
    const res = await apiFetch("/api/register", {
      method: "POST",
      body: JSON.stringify({ name, email, password })
    });

    if (res && res.success) {
      showToast("Account created successfully! Please log in.", "success");
      toggleAuthTab("login");
    }
  } catch (err) {
    // Handled in apiFetch
  }
}

function logout() {
  state.token = null;
  state.user = null;
  localStorage.removeItem("creditiq_token");
  localStorage.removeItem("creditiq_user");
  updateAuthUI();
  switchView("home");
  showToast("Logged out successfully.", "info");
}

function updateAuthUI() {
  const authBtn = document.getElementById("nav-auth-btn");
  const userBadge = document.getElementById("nav-user-badge");
  const userName = document.getElementById("user-name-display");
  const userRole = document.getElementById("user-role-display");
  const adminNav = document.getElementById("nav-admin-link");
  const myAppsNav = document.getElementById("nav-myapps-link");

  if (state.user) {
    if (authBtn) authBtn.style.display = "none";
    if (userBadge) userBadge.style.display = "flex";
    if (userName) userName.textContent = state.user.name;
    if (userRole) {
      userRole.textContent = state.user.role;
      userRole.className = `role-pill ${state.user.role === "admin" ? "role-admin" : "role-user"}`;
    }

    if (myAppsNav) myAppsNav.style.display = "flex";
    if (adminNav) adminNav.style.display = state.user.role === "admin" ? "flex" : "none";
  } else {
    if (authBtn) authBtn.style.display = "inline-flex";
    if (userBadge) userBadge.style.display = "none";
    if (myAppsNav) myAppsNav.style.display = "none";
    if (adminNav) adminNav.style.display = "none";
  }
}

/* ─── Navigation ───────────────────────────────────────────────────────────── */

function setupNavigation() {
  document.querySelectorAll("[data-target]").forEach(element => {
    element.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = element.getAttribute("data-target");
      switchView(targetView);
    });
  });
}

function switchView(viewName) {
  // Check auth requirement for restricted views
  if ((viewName === "apply" || viewName === "myapps" || viewName === "admin") && !state.user) {
    showToast("Please log in to access this page.", "info");
    openModal("auth-modal");
    return;
  }

  if (viewName === "admin" && state.user?.role !== "admin") {
    showToast("Admin access required.", "error");
    switchView("home");
    return;
  }

  state.currentView = viewName;

  // Update navbar active state
  document.querySelectorAll(".nav-link").forEach(link => {
    if (link.getAttribute("data-target") === viewName) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  // Toggle view visibility
  document.querySelectorAll(".view").forEach(view => {
    if (view.id === `view-${viewName}`) {
      view.classList.add("active");
    } else {
      view.classList.remove("active");
    }
  });

  // Load view data
  if (viewName === "myapps") {
    loadMyApplications();
  } else if (viewName === "admin") {
    loadAdminDashboard();
  }
}

/* ─── Forms ────────────────────────────────────────────────────────────────── */

function setupForms() {
  // Auth Tab Switcher
  const loginTabBtn = document.getElementById("tab-btn-login");
  const regTabBtn = document.getElementById("tab-btn-register");
  
  if (loginTabBtn && regTabBtn) {
    loginTabBtn.addEventListener("click", () => toggleAuthTab("login"));
    regTabBtn.addEventListener("click", () => toggleAuthTab("register"));
  }

  // Auth Form Submit
  const authForm = document.getElementById("auth-form");
  if (authForm) {
    authForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const mode = authForm.getAttribute("data-mode");
      const email = document.getElementById("auth-email").value.trim();
      const password = document.getElementById("auth-password").value;

      if (mode === "register") {
        const name = document.getElementById("auth-name").value.trim();
        registerUser(name, email, password);
      } else {
        loginUser(email, password);
      }
    });
  }

  // Loan Application Form Submit
  const loanForm = document.getElementById("loan-application-form");
  if (loanForm) {
    loanForm.addEventListener("submit", handleLoanApplicationSubmit);
  }
}

function toggleAuthTab(mode) {
  const authForm = document.getElementById("auth-form");
  const nameGroup = document.getElementById("group-auth-name");
  const submitBtn = document.getElementById("auth-submit-btn");
  const loginTab = document.getElementById("tab-btn-login");
  const regTab = document.getElementById("tab-btn-register");

  authForm.setAttribute("data-mode", mode);

  if (mode === "register") {
    nameGroup.style.display = "block";
    submitBtn.textContent = "Create Account";
    regTab.classList.add("active");
    loginTab.classList.remove("active");
  } else {
    nameGroup.style.display = "none";
    submitBtn.textContent = "Sign In";
    loginTab.classList.add("active");
    regTab.classList.remove("active");
  }
}

/* ─── Loan Application Submission ──────────────────────────────────────────── */

async function handleLoanApplicationSubmit(e) {
  e.preventDefault();

  if (!state.user) {
    showToast("Please log in or register an account to submit your loan application.", "info");
    openModal("auth-modal");
    return;
  }

  const formData = {
    age: parseInt(document.getElementById("input-age").value),
    employment_type: document.getElementById("input-employment").value,
    monthly_income: parseFloat(document.getElementById("input-income").value),
    existing_emis: parseFloat(document.getElementById("input-emis").value),
    loan_amount: parseFloat(document.getElementById("input-loanamount").value),
    loan_tenure: parseInt(document.getElementById("input-tenure").value),
    missed_payments: parseInt(document.getElementById("input-missedpmts").value),
    credit_utilization: parseFloat(document.getElementById("input-utilization").value),
    credit_history_years: parseFloat(document.getElementById("input-history").value)
  };

  const submitBtn = document.getElementById("btn-submit-loan");
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span>Analyzing Application...</span>`;

  try {
    const res = await apiFetch("/api/predict", {
      method: "POST",
      body: JSON.stringify(formData)
    });

    if (res && res.success) {
      showToast("Application analyzed successfully!", "success");
      renderPredictionResults(res);
    }
  } catch (err) {
    // Handled in apiFetch
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<span>Submit Application</span>`;
  }
}

function renderPredictionResults(data) {
  const resultCard = document.getElementById("prediction-results-card");
  resultCard.style.display = "block";
  resultCard.scrollIntoView({ behavior: "smooth" });

  // Credit Score Gauge
  const scoreVal = document.getElementById("res-credit-score");
  const scoreLabel = document.getElementById("res-score-label");
  scoreVal.textContent = data.credit_score;
  scoreLabel.textContent = data.score_label;

  let scoreClass = "score-good";
  if (data.credit_score >= 750) scoreClass = "score-excellent";
  else if (data.credit_score >= 650) scoreClass = "score-good";
  else if (data.credit_score >= 580) scoreClass = "score-fair";
  else scoreClass = "score-poor";
  scoreLabel.className = `score-badge ${scoreClass}`;

  // Risk & Probability
  document.getElementById("res-risk-category").textContent = data.risk_category;
  document.getElementById("res-probability-pct").textContent = data.probability_pct;
  document.getElementById("res-ml-recommendation").textContent = data.ml_recommendation;

  // Reason Codes
  const reasonsContainer = document.getElementById("res-reason-codes");
  reasonsContainer.innerHTML = (data.reason_codes || []).map(r => `
    <div style="background: rgba(30,41,59,0.5); padding: 0.5rem 0.8rem; border-radius: 8px; border-left: 3px solid var(--primary); font-size: 0.85rem; margin-bottom: 0.4rem;">
      ${r}
    </div>
  `).join("");

  // Recommendations
  const recsContainer = document.getElementById("res-recommendations");
  recsContainer.innerHTML = (data.recommendations || []).map(rec => `
    <div style="padding: 0.6rem; border-radius: 8px; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); margin-bottom: 0.5rem; font-size: 0.85rem;">
      💡 ${rec}
    </div>
  `).join("");

  // SHAP chart image or fallback text
  const shapImg = document.getElementById("res-shap-img");
  if (data.shap_chart_b64) {
    shapImg.src = `data:image/png;base64,${data.shap_chart_b64}`;
    shapImg.style.display = "block";
  } else {
    shapImg.style.display = "none";
  }
}

/* ─── My Applications ──────────────────────────────────────────────────────── */

async function loadMyApplications() {
  const container = document.getElementById("my-applications-list");
  container.innerHTML = `<div style="text-align:center; padding: 2rem; color: var(--text-muted);">Loading applications...</div>`;

  try {
    const res = await apiFetch("/api/my-applications");
    if (res && res.success) {
      state.applications = res.applications || [];
      renderMyApplicationsList(state.applications);
    }
  } catch (err) {
    container.innerHTML = `<div style="text-align:center; padding: 2rem; color: var(--accent-rose);">Failed to load applications.</div>`;
  }
}

function renderMyApplicationsList(apps) {
  const container = document.getElementById("my-applications-list");

  if (!apps.length) {
    container.innerHTML = `
      <div style="text-align:center; padding: 3rem; background: var(--bg-card); border-radius: var(--radius-lg); border: 1px solid var(--border-color);">
        <h3>No applications yet</h3>
        <p style="color: var(--text-muted); margin: 0.8rem 0 1.5rem 0;">Submit your first credit risk assessment application.</p>
        <button class="btn btn-primary" data-target="apply">Apply Now</button>
      </div>
    `;
    setupNavigation();
    return;
  }

  container.innerHTML = `
    <div class="table-container">
      <table class="custom-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Date</th>
            <th>Credit Score</th>
            <th>Risk Level</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          ${apps.map(a => `
            <tr>
              <td class="mono">#${a.id}</td>
              <td>${new Date(a.created_at || Date.now()).toLocaleDateString()}</td>
              <td class="mono" style="font-weight: 700;">${a.credit_score || 'N/A'}</td>
              <td><span class="badge" style="background: rgba(59,130,246,0.15); color: var(--primary);">${a.risk_category || 'Normal'}</span></td>
              <td><span class="badge badge-${(a.status || 'Pending').toLowerCase()}">${a.status || 'Pending'}</span></td>
              <td>
                <button class="btn btn-secondary btn-sm" onclick="viewApplicationDetail(${a.id})">View Details</button>
              </td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function viewApplicationDetail(appId) {
  try {
    const res = await apiFetch(`/api/application/${appId}`);
    if (res && res.success) {
      state.selectedApp = res;
      openModal("app-detail-modal");

      const body = document.getElementById("modal-app-detail-body");
      body.innerHTML = `
        <div style="margin-bottom: 1.5rem;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2>Application #${res.id}</h2>
            <span class="badge badge-${(res.status || 'Pending').toLowerCase()}">${res.status}</span>
          </div>
          <p style="color: var(--text-muted); font-size: 0.85rem;">Submitted on ${new Date(res.created_at).toLocaleString()}</p>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
          <div class="stat-box">
            <div class="stat-value">${res.credit_score}</div>
            <div class="stat-label">Credit Score (${res.score_label})</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">${(res.probability * 100).toFixed(1)}%</div>
            <div class="stat-label">Default Risk Prob</div>
          </div>
        </div>

        <div style="margin-bottom: 1rem;">
          <h4>ML Recommendation</h4>
          <p style="color: var(--text-main); font-weight: 600; font-size: 0.95rem;">${res.ml_recommendation}</p>
        </div>

        <div style="margin-bottom: 1rem;">
          <h4>Reason Codes</h4>
          <p style="color: var(--text-muted); font-size: 0.88rem;">${res.reason_codes || 'None'}</p>
        </div>

        ${res.decision_history?.length ? `
          <div style="margin-top: 1.5rem; border-top: 1px solid var(--border-color); padding-top: 1rem;">
            <h4>Decision History</h4>
            ${res.decision_history.map(h => `
              <div style="font-size: 0.82rem; margin-top: 0.5rem; color: var(--text-muted);">
                <strong>${h.status}</strong> by ${h.admin_email} on ${new Date(h.timestamp).toLocaleString()}
                ${h.note ? `<br><em>"${h.note}"</em>` : ''}
              </div>
            `).join("")}
          </div>
        ` : ''}
      `;
    }
  } catch (err) {
    // Handled in apiFetch
  }
}

/* ─── Admin Dashboard ──────────────────────────────────────────────────────── */

async function loadAdminDashboard() {
  try {
    const stats = await apiFetch("/api/admin/dashboard");
    if (stats && stats.success) {
      document.getElementById("admin-total-apps").textContent = stats.total_applications || 0;
      document.getElementById("admin-approved-apps").textContent = stats.approved_count || 0;
      document.getElementById("admin-pending-apps").textContent = stats.pending_count || 0;
      document.getElementById("admin-rejected-apps").textContent = stats.rejected_count || 0;
      document.getElementById("admin-avg-score").textContent = stats.avg_credit_score ? Math.round(stats.avg_credit_score) : 0;
    }

    const appsRes = await apiFetch("/api/admin/applications");
    if (appsRes && appsRes.success) {
      state.adminApplications = appsRes.applications || [];
      renderAdminTable(state.adminApplications);
    }
  } catch (err) {
    console.error("Admin dashboard load error:", err);
  }
}

function renderAdminTable(apps) {
  const container = document.getElementById("admin-applications-table");
  if (!container) return;

  container.innerHTML = `
    <div class="table-container">
      <table class="custom-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Applicant</th>
            <th>Credit Score</th>
            <th>Prob %</th>
            <th>Status</th>
            <th>Decision Action</th>
          </tr>
        </thead>
        <tbody>
          ${apps.map(a => `
            <tr>
              <td class="mono">#${a.id}</td>
              <td>
                <div style="font-weight:600;">${a.user_name || 'User'}</div>
                <div style="font-size:0.75rem; color: var(--text-muted);">${a.user_email || ''}</div>
              </td>
              <td class="mono" style="font-weight:700;">${a.credit_score}</td>
              <td class="mono">${(a.probability * 100).toFixed(1)}%</td>
              <td><span class="badge badge-${(a.status || 'Pending').toLowerCase()}">${a.status}</span></td>
              <td>
                <div style="display:flex; gap:0.4rem;">
                  <button class="btn btn-success btn-sm" onclick="makeAdminDecision(${a.id}, 'Approved')">Approve</button>
                  <button class="btn btn-danger btn-sm" onclick="makeAdminDecision(${a.id}, 'Rejected')">Reject</button>
                </div>
              </td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function makeAdminDecision(appId, status) {
  const note = prompt(`Enter admin decision note for #${appId} (${status}):`, `Verified by admin`);
  if (note === null) return;

  try {
    const res = await apiFetch(`/api/admin/decision/${appId}`, {
      method: "POST",
      body: JSON.stringify({ status, note })
    });

    if (res && res.success) {
      showToast(res.message, "success");
      loadAdminDashboard();
    }
  } catch (err) {
    // Handled in apiFetch
  }
}

/* ─── API Config Modal ─────────────────────────────────────────────────────── */

function initApiUrlInput() {
  const input = document.getElementById("api-base-url-input");
  if (input) {
    input.value = state.apiBase;
  }
}

function saveApiUrl() {
  const input = document.getElementById("api-base-url-input");
  let val = input.value.trim().replace(/\/$/, "");
  if (!val) val = DEFAULT_API_BASE;

  state.apiBase = val;
  localStorage.setItem("creditiq_api_base", val);
  closeModal("config-modal");
  showToast(`API URL updated to ${val}`, "success");
  checkApiHealth();
}

/* ─── Modal Helpers ───────────────────────────────────────────────────────── */

function openModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.classList.add("active");
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.classList.remove("active");
}

/* ─── Toast Notifications ─────────────────────────────────────────────────── */

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✅' : type === 'error' ? '⚠️' : 'ℹ️'}</span>
    <div>${message}</div>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
