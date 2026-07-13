"""
frontend/app.py
===============
Streamlit multi-page app — Credit Risk Assessment System v2

Pages (User):
  🏠 Home
  🔐 Login / Register
  📋 Apply for Loan
  📊 My Applications
  📄 Application Detail

Pages (Admin only):
  🛡 Review Applications
  📈 Analytics Dashboard
  🧪 Model Evaluation
  👥 User Management
"""

import sys, os, json, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

API = os.environ.get("API_URL", "http://127.0.0.1:5000")

# ═══════════════════════════════════════════════════════════════════
#  CSS — refined editorial-finance aesthetic
# ═══════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.4rem; max-width: 1080px; }
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* App background */
.stApp {
  background: #0c0f1a;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(59,130,246,.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139,92,246,.06) 0%, transparent 60%);
  color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #0f1629;
  border-right: 1px solid rgba(59,130,246,.2);
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: .88rem !important; }

/* Logo */
.logo-text {
  font-size: 1.55rem; font-weight: 800; letter-spacing: -1px;
  background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.logo-sub { font-size: .7rem; color: #475569; text-transform: uppercase; letter-spacing: .1em; }

/* Cards */
.card {
  background: rgba(30,41,59,.7);
  border: 1px solid rgba(71,85,105,.5);
  border-radius: 14px; padding: 20px 22px;
  margin-bottom: 14px; backdrop-filter: blur(4px);
  transition: border-color .2s, box-shadow .2s;
}
.card:hover { border-color: rgba(96,165,250,.4); box-shadow: 0 0 18px rgba(96,165,250,.08); }
.card-sm {
  background: rgba(30,41,59,.5);
  border: 1px solid rgba(71,85,105,.4);
  border-radius: 10px; padding: 14px 16px;
}

/* KPI cards */
.kpi {
  background: rgba(15,23,42,.8);
  border: 1px solid rgba(71,85,105,.5);
  border-radius: 12px; padding: 18px;
  text-align: center;
}
.kpi-num  { font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; }
.kpi-lbl  { font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing: .1em; margin-top: 2px; }

/* Status badges */
.status-pending  { display:inline-block; background:#422006; color:#fbbf24; border:1.5px solid #d97706; border-radius:20px; padding:5px 16px; font-weight:700; font-size:.82rem; }
.status-approved { display:inline-block; background:#052e16; color:#4ade80; border:1.5px solid #16a34a; border-radius:20px; padding:5px 16px; font-weight:700; font-size:.82rem; }
.status-rejected { display:inline-block; background:#450a0a; color:#f87171; border:1.5px solid #dc2626; border-radius:20px; padding:5px 16px; font-weight:700; font-size:.82rem; }

/* Risk pills */
.risk-low    { display:inline-block; background:#052e16; color:#4ade80; border-radius:4px; padding:3px 10px; font-size:.75rem; font-weight:600; }
.risk-medium { display:inline-block; background:#422006; color:#fbbf24; border-radius:4px; padding:3px 10px; font-size:.75rem; font-weight:600; }
.risk-high   { display:inline-block; background:#450a0a; color:#f87171; border-radius:4px; padding:3px 10px; font-size:.75rem; font-weight:600; }

/* Priority tags */
.tag-high   { background:#450a0a; color:#f87171;  border-radius:4px; padding:2px 8px; font-size:.72rem; font-weight:700; }
.tag-medium { background:#422006; color:#fbbf24;  border-radius:4px; padding:2px 8px; font-size:.72rem; font-weight:700; }
.tag-low    { background:#052e16; color:#4ade80;  border-radius:4px; padding:2px 8px; font-size:.72rem; font-weight:700; }
.tag-info   { background:#0c1a3a; color:#93c5fd;  border-radius:4px; padding:2px 8px; font-size:.72rem; font-weight:700; }

/* Alert banners */
.alert-ok   { background:#052e16; border-left:4px solid #16a34a; padding:12px 16px; border-radius:0 8px 8px 0; color:#4ade80;  margin:8px 0; }
.alert-warn { background:#422006; border-left:4px solid #d97706; padding:12px 16px; border-radius:0 8px 8px 0; color:#fbbf24;  margin:8px 0; }
.alert-err  { background:#450a0a; border-left:4px solid #dc2626; padding:12px 16px; border-radius:0 8px 8px 0; color:#f87171;  margin:8px 0; }
.alert-info { background:#0c1a3a; border-left:4px solid #3b82f6; padding:12px 16px; border-radius:0 8px 8px 0; color:#93c5fd;  margin:8px 0; }

/* Section label */
.sec-label {
  font-size:.68rem; font-weight:700; color:#60a5fa;
  text-transform:uppercase; letter-spacing:.12em; margin-bottom:6px;
}

/* Score card */
.score-card {
  background: linear-gradient(135deg, rgba(15,52,96,.8), rgba(30,41,59,.8));
  border: 1px solid rgba(59,130,246,.4);
  border-radius: 16px; padding: 22px; text-align: center;
}
.score-num  { font-size: 3rem; font-weight: 800; letter-spacing: -2px; }
.score-bar  { height: 6px; background: #0f172a; border-radius: 3px; overflow:hidden; margin-top:10px; }
.score-fill { height: 100%; background: linear-gradient(90deg,#ef4444 0%,#eab308 45%,#22c55e 100%); border-radius:3px; }

/* Workflow steps */
.step { display:inline-flex; align-items:center; gap:8px; }
.step-num {
  width:28px; height:28px; border-radius:50%;
  background:rgba(59,130,246,.2); border:1.5px solid #3b82f6;
  display:inline-flex; align-items:center; justify-content:center;
  font-size:.78rem; font-weight:700; color:#60a5fa;
}

/* Buttons */
.stButton>button {
  background: linear-gradient(135deg, #2563eb, #4f46e5);
  color: white; border: none; border-radius: 9px;
  padding: 10px 26px; font-weight: 600; font-size:.88rem;
  transition: opacity .2s; font-family: 'Sora', sans-serif;
}
.stButton>button:hover { opacity: .85; }

/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>div {
  background: rgba(30,41,59,.8) !important;
  border: 1px solid rgba(71,85,105,.6) !important;
  color: #e2e8f0 !important; border-radius: 8px !important;
  font-family: 'Sora', sans-serif !important;
}
.stSlider { color: #94a3b8 !important; }

/* Metrics */
[data-testid="stMetricValue"] { color: #60a5fa !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: .8rem !important; }

/* Table */
.stDataFrame { background: rgba(30,41,59,.5) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(30,41,59,.5); border-radius: 10px 10px 0 0; gap:2px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; font-family: 'Sora',sans-serif; }
.stTabs [aria-selected="true"] { color: #60a5fa !important; }

h1,h2,h3 { color: #e2e8f0 !important; font-family: 'Sora', sans-serif !important; }
p, li { font-size: .9rem; color: #94a3b8; }
</style>"""


# ═══════════════════════════════════════════════════════════════════
#  Session helpers
# ═══════════════════════════════════════════════════════════════════
tok   = lambda: st.session_state.get("token")
usr   = lambda: st.session_state.get("user", {})
is_admin = lambda: usr().get("role") == "admin"
auth_h   = lambda: {"Authorization": f"Bearer {tok()}"}


def api_call(method, path, **kwargs):
    """Safe API call — returns (data_or_None, error_string_or_None)."""
    try:
        kwargs.setdefault("timeout", 15)
        fn = getattr(requests, method)
        r  = fn(f"{API}{path}", **kwargs)
        try:
            body = r.json()
        except Exception:
            body = {}

        if r.status_code >= 400:
            msg = body.get("error") or body.get("message") or f"HTTP {r.status_code}"
            return None, msg
        return body, None
    except requests.ConnectionError:
        return None, "Cannot reach the backend. Make sure Flask is running on port 5000."
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════════
#  UI helpers
# ═══════════════════════════════════════════════════════════════════
def alert(msg, kind="info"):
    st.markdown(f'<div class="alert-{kind}">{msg}</div>', unsafe_allow_html=True)


def status_badge(status):
    cls = {"Pending": "status-pending", "Approved": "status-approved",
           "Rejected": "status-rejected"}.get(status, "status-pending")
    st.markdown(f'<span class="{cls}">{status}</span>', unsafe_allow_html=True)


def risk_badge(risk):
    cls = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(risk, "risk-medium")
    st.markdown(f'<span class="{cls}">{risk} RISK</span>', unsafe_allow_html=True)


def gauge_chart(prob: float, title="Risk Probability"):
    pct = prob * 100
    clr = "#22c55e" if pct < 40 else "#eab308" if pct < 70 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"color": clr, "size": 40, "family": "Sora"}},
        title={"text": title, "font": {"color": "#64748b", "size": 12}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#334155"},
            "bar":   {"color": clr, "thickness": 0.24},
            "bgcolor": "rgba(30,41,59,.8)",
            "steps": [
                {"range": [0,  40], "color": "rgba(5,46,22,.4)"},
                {"range": [40, 70], "color": "rgba(66,32,6,.4)"},
                {"range": [70,100], "color": "rgba(69,10,10,.4)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2},
                          "thickness": 0.75, "value": pct},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=30,b=5,l=10,r=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig


def score_card(score, label):
    pct  = (score - 300) / 6
    clrs = {"Very Poor":"#ef4444","Poor":"#f97316","Fair":"#eab308",
            "Good":"#3b82f6","Very Good":"#8b5cf6","Excellent":"#22c55e"}
    c    = clrs.get(label, "#60a5fa")
    st.markdown(f"""
    <div class="score-card">
      <div class="sec-label">Credit Score</div>
      <div class="score-num" style="color:{c}">{score}</div>
      <div style="color:#64748b;font-size:.88rem">{label}</div>
      <div class="score-bar"><div class="score-fill" style="width:{pct:.0f}%"></div></div>
      <div style="display:flex;justify-content:space-between;color:#475569;font-size:.7rem;margin-top:4px">
        <span>300 — Poor</span><span>900 — Excellent</span>
      </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style="text-align:center;padding:30px 0 20px">
      <div class="logo-text" style="font-size:2.6rem">CreditIQ</div>
      <div style="color:#64748b;margin-top:6px;font-size:1rem">
        AI-Powered Credit Risk Assessment System
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    items = [
        ("🤖","CNN + BiLSTM + TabTransformer","Three-model ensemble for accurate risk prediction"),
        ("📊","Credit Score Engine","Automatic 300–900 score — no user input needed"),
        ("🔍","SHAP Explainability","Every prediction backed by feature-level explanation"),
        ("🛡","RBAC Workflow","Separate user and admin roles with secure access control"),
    ]
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],items):
        with col:
            st.markdown(f'<div class="card" style="text-align:center;min-height:130px">'
                        f'<div style="font-size:1.8rem;margin-bottom:8px">{icon}</div>'
                        f'<strong style="color:#e2e8f0;font-size:.88rem">{title}</strong>'
                        f'<br><small style="color:#64748b">{desc}</small></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-top:20px">Application Workflow</div>', unsafe_allow_html=True)
    wf_cols = st.columns(5)
    steps = [("1","Register","Create your account"),("2","Apply","Fill in loan details"),
             ("3","AI Predicts","Risk + credit score computed"),("4","Admin Reviews","Approves or rejects"),
             ("5","Get Result","View decision + advice")]
    for col,(n,t,d) in zip(wf_cols,steps):
        with col:
            st.markdown(f'<div class="card-sm" style="text-align:center">'
                        f'<div class="step-num" style="margin:0 auto 8px">{n}</div>'
                        f'<strong style="font-size:.82rem;color:#e2e8f0">{t}</strong>'
                        f'<br><small style="color:#64748b">{d}</small></div>',
                        unsafe_allow_html=True)

    if not tok():
        alert("👈 Use the sidebar to Login or Register to get started.", "info")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: AUTH
# ═══════════════════════════════════════════════════════════════════
def page_auth():
    st.markdown("## 🔐 Account Access")
    login_tab, reg_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        with st.form("lf"):
            em = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign In", use_container_width=True):
                data, err_msg = api_call("post", "/api/login",
                                         json={"email": em, "password": pw})
                if data:
                    st.session_state.update({"token": data["token"], "user": data["user"]})
                    alert("✅ Signed in successfully!", "ok")
                    st.rerun()
                elif err_msg:
                    alert(f"❌ {err_msg}", "err")

    with reg_tab:
        with st.form("rf"):
            nm = st.text_input("Full Name")
            em = st.text_input("Email")
            pw = st.text_input("Password (min 6 chars)", type="password")
            if st.form_submit_button("Create Account", use_container_width=True):
                data, err_msg = api_call("post", "/api/register",
                                         json={"name": nm, "email": em, "password": pw})
                if data:
                    alert("✅ Account created! Please sign in.", "ok")
                elif err_msg:
                    alert(f"❌ {err_msg}", "err")

    st.markdown("""<div class="card-sm" style="margin-top:24px">
    <div class="sec-label">Demo Credentials</div>
    <b style="color:#e2e8f0">Admin login:</b> <code>admin@bank.com</code> / <code>admin123</code><br>
    <small style="color:#64748b">Register your own user account using the tab above.</small>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: APPLY
# ═══════════════════════════════════════════════════════════════════
def page_apply():
    if not tok():
        alert("Please sign in to submit a loan application.", "warn"); return

    st.markdown("## 📋 Loan Application")
    alert("ℹ️ You do not need to enter a credit score. Our system calculates it automatically.", "info")

    with st.form("loan_form"):
        st.markdown('<div class="sec-label">Personal & Employment</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: age  = st.number_input("Age", 18, 80, 30)
        with c2: emp  = st.selectbox("Employment Type", ["salaried","self-employed","unemployed"])
        with c3: inc  = st.number_input("Monthly Income (₹)", 0, 10_000_000, 55000, step=1000)

        st.markdown('<div class="sec-label">Loan Request</div>', unsafe_allow_html=True)
        c4,c5 = st.columns(2)
        with c4: loan = st.number_input("Loan Amount (₹)", 10000, 50_000_000, 500000, step=10000)
        with c5: ten  = st.number_input("Repayment Tenure (months)", 3, 360, 60)

        st.markdown('<div class="sec-label">Financial Health</div>', unsafe_allow_html=True)
        c6,c7,c8,c9 = st.columns(4)
        with c6: emis = st.number_input("Existing Monthly EMIs (₹)", 0, 1_000_000, 8000, step=500)
        with c7: miss = st.number_input("Missed Payments (last 12m)", 0, 12, 0)
        with c8: hist = st.number_input("Credit History (years)", 0.0, 40.0, 3.0, step=0.5)
        with c9: util = st.slider("Credit Utilization %", 0, 100, 30)

        dti_v = emis / inc if inc > 0 else 0
        alert(f"{'✅' if dti_v<0.4 else '⚠️'} Current Debt-to-Income Ratio: <b>{dti_v:.1%}</b>"
              f" — {'within acceptable range' if dti_v<0.4 else 'above recommended 40%'}",
              "ok" if dti_v < 0.4 else "warn")

        submitted = st.form_submit_button("🚀 Submit Application", use_container_width=True)

    if submitted:
        payload = {"age": int(age), "employment_type": emp,
                   "monthly_income": float(inc), "existing_emis": float(emis),
                   "loan_amount": float(loan), "loan_tenure": int(ten),
                   "missed_payments": int(miss), "credit_utilization": float(util),
                   "credit_history_years": float(hist)}
        with st.spinner("🔄 Analysing with AI models…"):
            data, err_msg = api_call("post", "/api/predict", json=payload, headers=auth_h())
        if data:
            st.session_state["result"] = data
            st.session_state["inputs"] = payload
            alert("✅ Application submitted! View results below or in 'My Applications'.", "ok")
            _show_result_summary(data)
        elif err_msg:
            alert(f"❌ {err_msg}", "err")


def _show_result_summary(data):
    """Quick inline summary after submission."""
    st.markdown("---")
    st.markdown("### Quick Summary")
    c1,c2,c3 = st.columns([1.2,2,2])
    with c1:
        st.markdown('<div class="sec-label">Status</div>', unsafe_allow_html=True)
        status_badge("Pending")
        st.markdown("<br>", unsafe_allow_html=True)
        risk_badge(data.get("risk_category","—"))
        st.markdown(f"<small style='color:#64748b'>App #{data.get('application_id','—')}</small>",
                    unsafe_allow_html=True)
    with c2:
        st.plotly_chart(gauge_chart(data.get("probability",0)), use_container_width=True)
    with c3:
        score_card(data.get("credit_score",300), data.get("score_label","—"))

    alert(f"🤖 <b>AI Recommendation:</b> {data.get('ml_recommendation','—')}<br>"
          f"<small>The final decision will be made by an admin after review.</small>", "info")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: MY APPLICATIONS
# ═══════════════════════════════════════════════════════════════════
def page_my_apps():
    if not tok():
        alert("Please sign in to view your applications.", "warn"); return

    st.markdown("## 📊 My Applications")

    data, err_msg = api_call("get", "/api/my-applications", headers=auth_h())
    if err_msg:
        alert(f"❌ {err_msg}", "err"); return

    apps = data.get("applications", [])
    if not apps:
        alert("You haven't submitted any applications yet. Go to 'Apply for Loan' to get started.", "info")
        return

    # Status filter tabs
    t_all, t_pend, t_appr, t_rej = st.tabs(
        [f"All ({len(apps)})",
         f"⏳ Pending ({sum(1 for a in apps if a.get('status')=='Pending')})",
         f"✅ Approved ({sum(1 for a in apps if a.get('status')=='Approved')})",
         f"❌ Rejected ({sum(1 for a in apps if a.get('status')=='Rejected')})"]
    )

    def render_apps_table(filtered):
        if not filtered:
            st.info("No applications in this category.")
            return
        df = pd.DataFrame(filtered)
        show_cols = [c for c in ["id","credit_score","probability","risk_category",
                                  "ml_recommendation","status","created_at"] if c in df]
        if "probability" in df.columns:
            df["probability"] = df["probability"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
        st.markdown('<div class="sec-label">Click an application to see full details</div>',
                    unsafe_allow_html=True)
        app_id = st.number_input("Enter Application ID for details", min_value=1, step=1,
                                  key=f"aid_{filtered[0]['id']}")
        if st.button("View Details", key=f"btn_{filtered[0]['id']}"):
            st.session_state["view_app_id"] = int(app_id)
            st.rerun()

    with t_all:  render_apps_table(apps)
    with t_pend: render_apps_table([a for a in apps if a.get("status")=="Pending"])
    with t_appr: render_apps_table([a for a in apps if a.get("status")=="Approved"])
    with t_rej:  render_apps_table([a for a in apps if a.get("status")=="Rejected"])

    # Show detail if selected
    if "view_app_id" in st.session_state:
        _show_app_detail(st.session_state["view_app_id"])


def _show_app_detail(app_id):
    st.markdown("---")
    st.markdown(f"### Application #{app_id} — Full Detail")
    data, err_msg = api_call("get", f"/api/application/{app_id}", headers=auth_h())
    if err_msg:
        alert(f"❌ {err_msg}", "err"); return

    c1,c2,c3 = st.columns([1.2,2,2])
    with c1:
        st.markdown('<div class="sec-label">Admin Decision</div>', unsafe_allow_html=True)
        status_badge(data.get("status","Pending"))
        st.markdown("<br>", unsafe_allow_html=True)
        risk_badge(data.get("risk_category","—"))
        st.markdown(f'<small style="color:#64748b">ML Rec: {data.get("ml_recommendation","—")}</small>',
                    unsafe_allow_html=True)
    with c2:
        st.plotly_chart(gauge_chart(data.get("probability",0)), use_container_width=True)
    with c3:
        score_card(data.get("credit_score",300), data.get("score_label","—"))

    # Score breakdown
    bd = data.get("score_breakdown") or {}
    if bd:
        st.markdown('<div class="sec-label">Credit Score Breakdown</div>', unsafe_allow_html=True)
        for c,(k,v) in zip(st.columns(len(bd)), bd.items()):
            with c: st.metric(k.replace("_"," ").title(), f"{v}/100")

    # Model predictions
    mp = data.get("model_predictions") or {}
    if mp:
        st.markdown('<div class="sec-label">Individual Model Probabilities</div>', unsafe_allow_html=True)
        for col,(k,v) in zip(st.columns(len(mp)), mp.items()):
            with col:
                lbl = {"cnn":"CNN","lstm":"BiLSTM","tab":"TabTransformer","ensemble":"Ensemble ★"}.get(k,k)
                st.metric(lbl, f"{v*100:.1f}%")

    # SHAP explanation
    st.markdown('<div class="sec-label">AI Explanation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{data.get("shap_explanation","—")}</div>',
                unsafe_allow_html=True)
    b64 = data.get("shap_chart_b64")
    if b64:
        st.image(base64.b64decode(b64), caption="Feature Impact (SHAP)", use_column_width=True)

    # Reason codes
    rc = data.get("reason_codes","")
    if rc:
        st.markdown('<div class="sec-label">Risk Factors</div>', unsafe_allow_html=True)
        for r in rc.split(";"):
            if r.strip(): alert(f"⚠ {r.strip()}", "warn")

    # Recommendations
    recs = data.get("recommendations",[])
    if isinstance(recs, str):
        try: recs = json.loads(recs)
        except: recs = []
    if recs:
        st.markdown('<div class="sec-label">Personalised Recommendations</div>', unsafe_allow_html=True)
        for r in recs:
            if not isinstance(r, dict): continue
            p = r.get("priority","medium")
            tag_cls = {"high":"tag-high","medium":"tag-medium","low":"tag-low","info":"tag-info"}.get(p,"tag-info")
            st.markdown(
                f'<div class="card">'
                f'<span class="{tag_cls}">{p.upper()}</span>&nbsp;'
                f'<strong style="color:#e2e8f0">{r.get("title","")}</strong><br>'
                f'<span style="color:#94a3b8;font-size:.87rem">{r.get("detail","")}</span>'
                + (f'<br><i style="color:#60a5fa;font-size:.8rem">📈 {r["impact"]}</i>' if r.get("impact") else "")
                + '</div>', unsafe_allow_html=True)

    # Decision history
    hist = data.get("decision_history",[])
    if hist:
        st.markdown('<div class="sec-label">Decision History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)

    # PDF download
    st.markdown('<div class="sec-label">Download Report</div>', unsafe_allow_html=True)
    if st.button("📄 Generate PDF Report"):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from reports.pdf_report import generate_pdf
            inputs = data.get("input_json") or {}
            if isinstance(inputs, str):
                try: inputs = json.loads(inputs)
                except: inputs = {}
            pdf_bytes = generate_pdf(data, inputs)
            st.download_button("⬇️ Download PDF", data=pdf_bytes,
                               file_name=f"credit_report_{app_id}.pdf",
                               mime="application/pdf")
        except Exception as e:
            alert(f"PDF error: {e}", "err")


# ═══════════════════════════════════════════════════════════════════
#  ADMIN: REVIEW APPLICATIONS
# ═══════════════════════════════════════════════════════════════════
def page_admin_review():
    if not is_admin():
        alert("🛡 Admin access required.", "err"); return

    st.markdown("## 🛡 Review Applications")

    # Filters
    c1,c2 = st.columns(2)
    with c1: f_status = st.selectbox("Filter by Status", ["All","Pending","Approved","Rejected"])
    with c2: f_risk   = st.selectbox("Filter by Risk",   ["All","LOW","MEDIUM","HIGH"])

    params = {}
    if f_status != "All": params["status"] = f_status
    if f_risk   != "All": params["risk"]   = f_risk

    data, err_msg = api_call("get", "/api/admin/applications", params=params, headers=auth_h())
    if err_msg:
        alert(f"❌ {err_msg}", "err"); return

    apps = data.get("applications", [])
    total = data.get("total", 0)

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    pending  = sum(1 for a in apps if a.get("status")=="Pending")
    approved = sum(1 for a in apps if a.get("status")=="Approved")
    rejected = sum(1 for a in apps if a.get("status")=="Rejected")
    k1.metric("Showing", total)
    k2.metric("⏳ Pending",  pending)
    k3.metric("✅ Approved", approved)
    k4.metric("❌ Rejected", rejected)

    if not apps:
        alert("No applications found with these filters.", "info"); return

    # Pending applications — approve/reject buttons
    pending_apps = [a for a in apps if a.get("status") == "Pending"]
    if pending_apps:
        st.markdown('<div class="sec-label">Pending — Awaiting Review</div>', unsafe_allow_html=True)
        for a in pending_apps:
            with st.expander(
                f"App #{a['id']} | {a.get('user_name','—')} | "
                f"Risk: {a.get('risk_category','—')} | "
                f"Prob: {a.get('probability',0)*100:.1f}% | "
                f"ML Rec: {a.get('ml_recommendation','—')}"
            ):
                c1,c2,c3 = st.columns([2,1,1])
                with c1:
                    st.markdown(f"**Applicant:** {a.get('user_email','—')}")
                    st.markdown(f"**Credit Score:** {a.get('credit_score','—')} ({a.get('score_label','—')})")
                    rc = a.get("reason_codes","")
                    if rc:
                        for r in rc.split(";"):
                            if r.strip():
                                st.markdown(f'<span style="color:#fbbf24;font-size:.82rem">⚠ {r.strip()}</span>',
                                            unsafe_allow_html=True)

                note_key = f"note_{a['id']}"
                st.text_input("Review Note (optional)", key=note_key,
                              placeholder="e.g. Income verified, DTI acceptable")

                col_approve, col_reject, col_pending = st.columns(3)
                with col_approve:
                    if st.button(f"✅ Approve #{a['id']}", key=f"app_{a['id']}"):
                        note = st.session_state.get(note_key, "")
                        d, e = api_call("post", f"/api/admin/decision/{a['id']}",
                                        json={"status":"Approved","note":note}, headers=auth_h())
                        alert(d["message"] if d else f"❌ {e}", "ok" if d else "err")
                        st.rerun()
                with col_reject:
                    if st.button(f"❌ Reject #{a['id']}", key=f"rej_{a['id']}"):
                        note = st.session_state.get(note_key, "")
                        d, e = api_call("post", f"/api/admin/decision/{a['id']}",
                                        json={"status":"Rejected","note":note}, headers=auth_h())
                        alert(d["message"] if d else f"❌ {e}", "ok" if d else "err")
                        st.rerun()

    # All applications table
    st.markdown('<div class="sec-label">All Applications</div>', unsafe_allow_html=True)
    df = pd.DataFrame(apps)
    show = [c for c in ["id","user_name","user_email","credit_score","probability",
                         "risk_category","ml_recommendation","status","reviewed_by","created_at"] if c in df]
    if "probability" in df.columns:
        df["probability"] = df["probability"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(df[show], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  ADMIN: ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def page_analytics():
    if not is_admin():
        alert("🛡 Admin access required.", "err"); return

    st.markdown("## 📈 Analytics Dashboard")

    dash, err_msg = api_call("get", "/api/admin/dashboard", headers=auth_h())
    if err_msg:
        alert(f"❌ {err_msg}", "err"); return

    # KPI row
    k = st.columns(6)
    kpis = [
        ("Total Applications", dash["total"],       "#60a5fa"),
        ("⏳ Pending",          dash["pending"],      "#fbbf24"),
        ("✅ Approved",          dash["approved"],     "#4ade80"),
        ("❌ Rejected",          dash["rejected"],     "#f87171"),
        ("🔴 High Risk",         dash["high_risk"],    "#f87171"),
        ("👥 Users",             dash["total_users"],  "#a78bfa"),
    ]
    for col,(lbl,val,clr) in zip(k, kpis):
        with col:
            st.markdown(f'<div class="kpi">'
                        f'<div class="kpi-num" style="color:{clr}">{val}</div>'
                        f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    # Model metrics
    mm = dash.get("model_metrics")
    if mm and isinstance(mm.get("accuracy"), float):
        st.markdown("---")
        st.markdown('<div class="sec-label">Ensemble Model Performance</div>', unsafe_allow_html=True)
        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        for col,k_name in zip([mc1,mc2,mc3,mc4,mc5],["accuracy","precision","recall","f1","auc"]):
            col.metric(k_name.title(), mm.get(k_name,"—"))

    st.markdown("---")
    apps_data, _ = api_call("get", "/api/admin/applications", headers=auth_h())
    apps = apps_data.get("applications", []) if apps_data else []

    if not apps:
        alert("No application data available yet.", "info"); return

    df = pd.DataFrame(apps)

    c1,c2 = st.columns(2)
    with c1:
        dec_df = df["status"].value_counts().reset_index()
        dec_df.columns = ["Status","Count"]
        fig = px.pie(dec_df, values="Count", names="Status", hole=0.45,
                     color="Status",
                     color_discrete_map={"Approved":"#22c55e","Pending":"#eab308","Rejected":"#ef4444"},
                     title="Application Status Distribution")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=340,
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        risk_df = df["risk_category"].value_counts().reset_index()
        risk_df.columns = ["Risk","Count"]
        fig2 = px.bar(risk_df, x="Risk", y="Count", color="Risk",
                      color_discrete_map={"LOW":"#22c55e","MEDIUM":"#eab308","HIGH":"#ef4444"},
                      title="Risk Level Distribution")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly trend
    trend = dash.get("monthly_trend", [])
    if trend:
        t_df = pd.DataFrame(trend)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=t_df["month"], y=t_df["total"],   name="Total",
                                  line=dict(color="#60a5fa",width=2.5),fill="tozeroy",
                                  fillcolor="rgba(96,165,250,.08)"))
        fig3.add_trace(go.Scatter(x=t_df["month"], y=t_df["approved"], name="Approved",
                                  line=dict(color="#4ade80",width=2)))
        fig3.add_trace(go.Scatter(x=t_df["month"], y=t_df["rejected"], name="Rejected",
                                  line=dict(color="#f87171",width=2)))
        fig3.update_layout(title="Monthly Application Trend",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", height=320,
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        if "credit_score" in df.columns:
            fig4 = px.histogram(df, x="credit_score", nbins=30,
                                color_discrete_sequence=["#8b5cf6"],
                                title="Credit Score Distribution")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=300)
            st.plotly_chart(fig4, use_container_width=True)
    with c4:
        if "probability" in df.columns:
            prob_col = df["probability"].copy()
            if prob_col.dtype == object:
                prob_col = prob_col.str.replace("%","").astype(float) / 100
            fig5 = px.box(df, x="status", y=prob_col,
                          color="status",
                          color_discrete_map={"Approved":"#22c55e","Pending":"#eab308","Rejected":"#ef4444"},
                          title="Probability Distribution by Status")
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", height=300, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
#  ADMIN: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════
def page_model_eval():
    if not is_admin():
        alert("🛡 Admin access required.", "err"); return

    st.markdown("## 🧪 Model Evaluation")

    comp_data, err_msg = api_call("get", "/api/model-comparison", headers=auth_h())
    if err_msg:
        alert(f"❌ {err_msg}", "err"); return

    if "message" in comp_data:
        alert(f"⚠ {comp_data['message']}", "warn")
        if st.button("▶️ Train Models Now (may take 5–10 min)"):
            with st.spinner("Training…"):
                r, e = api_call("post", "/api/admin/train", headers=auth_h())
            if r:  alert(f"✅ {r.get('message','Done')}", "ok")
            elif e: alert(f"❌ {e}", "err")
        return

    # Metrics table
    st.markdown('<div class="sec-label">Per-Model Metrics</div>', unsafe_allow_html=True)
    rows = []
    for name, m in comp_data.items():
        rows.append({"Model": name, "Accuracy": m.get("accuracy","—"),
                     "Precision": m.get("precision","—"), "Recall": m.get("recall","—"),
                     "F1": m.get("f1","—"), "AUC-ROC": m.get("auc","—"),
                     "TP": m.get("tp","—"), "FP": m.get("fp","—"),
                     "TN": m.get("tn","—"), "FN": m.get("fn","—")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Confusion matrices
    st.markdown('<div class="sec-label">Confusion Matrices</div>', unsafe_allow_html=True)
    cms_cols = st.columns(len(comp_data))
    for col,(name, m) in zip(cms_cols, comp_data.items()):
        with col:
            tn,fp,fn,tp = m.get("tn",0),m.get("fp",0),m.get("fn",0),m.get("tp",0)
            fig = px.imshow([[tn,fp],[fn,tp]], text_auto=True,
                             x=["Pred:Low","Pred:High"], y=["Act:Low","Act:High"],
                             color_continuous_scale="Blues", title=name)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0",height=220,
                              margin=dict(t=30,b=5,l=5,r=5))
            st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    metrics = ["accuracy","precision","recall","f1","auc"]
    model_names = [n for n in comp_data if n != "Ensemble"]
    fig_r = go.Figure()
    for mn, clr in zip(model_names, ["#60a5fa","#4ade80","#fbbf24"]):
        vals = [comp_data[mn].get(m,0) for m in metrics]
        fig_r.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=metrics+[metrics[0]], name=mn,
            line_color=clr, fill="toself",
            fillcolor=f"rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},.1)",
        ))
    fig_r.update_layout(
        polar=dict(bgcolor="rgba(30,41,59,.5)",
                   radialaxis=dict(visible=True, range=[0,1], color="#475569"),
                   angularaxis=dict(color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
        title="Model Comparison Radar", height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Evaluation plot image
    img_path = os.path.join(os.path.dirname(__file__), "..", "models", "evaluation_plots.png")
    if os.path.exists(img_path):
        st.markdown('<div class="sec-label">Training Evaluation Plots</div>', unsafe_allow_html=True)
        st.image(img_path, use_column_width=True)

    # Retrain button
    st.markdown("---")
    if st.button("🔄 Retrain Models"):
        with st.spinner("Retraining… this takes a few minutes."):
            r, e = api_call("post", "/api/admin/train", headers=auth_h())
        if r:  alert(f"✅ {r.get('message','Done')}", "ok")
        elif e: alert(f"❌ {e}", "err")


# ═══════════════════════════════════════════════════════════════════
#  ADMIN: USER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
def page_users():
    if not is_admin():
        alert("🛡 Admin access required.", "err"); return

    st.markdown("## 👥 User Management")
    data, err_msg = api_call("get", "/api/admin/users", headers=auth_h())
    if err_msg:
        alert(f"❌ {err_msg}", "err"); return

    users = data.get("users", [])
    st.metric("Total Registered Users", len(users))
    if users:
        st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  NAVIGATION
# ═══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="CreditIQ", page_icon="🏦",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="logo-text">CreditIQ</div>'
                    '<div class="logo-sub">Credit Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown("---")

        if tok() and usr():
            u = usr()
            role_badge = "🛡 Admin" if is_admin() else "👤 User"
            st.markdown(f"""
            <div style="background:rgba(15,22,41,.8);border:1px solid rgba(59,130,246,.2);
                        border-radius:10px;padding:12px;margin-bottom:12px">
              <div style="font-weight:600;color:#e2e8f0">{u['name']}</div>
              <div style="font-size:.75rem;color:#64748b">{u['email']}</div>
              <div style="margin-top:6px">
                <span style="background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.3);
                             border-radius:12px;padding:2px 10px;font-size:.72rem">{role_badge}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        # Build page list based on role
        if not tok():
            pages = ["🏠 Home", "🔐 Login / Register"]
        elif is_admin():
            pages = ["🏠 Home", "📋 Apply for Loan", "📊 My Applications",
                     "──── Admin ────",
                     "🛡 Review Applications", "📈 Analytics Dashboard",
                     "🧪 Model Evaluation", "👥 User Management",
                     "🚪 Logout"]
        else:
            pages = ["🏠 Home", "📋 Apply for Loan", "📊 My Applications", "🚪 Logout"]

        # Filter dividers from selectable pages
        selectable = [p for p in pages if not p.startswith("──")]
        choice = st.radio("", pages, label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<small style="color:#334155">CreditIQ v2.0<br>Final Year Project</small>',
                    unsafe_allow_html=True)

    # Logout
    if choice == "🚪 Logout":
        for k in ["token","user","result","inputs","view_app_id"]:
            st.session_state.pop(k, None)
        st.rerun()

    # Dispatch
    dispatch = {
        "🏠 Home":              page_home,
        "🔐 Login / Register":  page_auth,
        "📋 Apply for Loan":    page_apply,
        "📊 My Applications":   page_my_apps,
        "🛡 Review Applications": page_admin_review,
        "📈 Analytics Dashboard": page_analytics,
        "🧪 Model Evaluation":  page_model_eval,
        "👥 User Management":   page_users,
    }
    dispatch.get(choice, page_home)()


if __name__ == "__main__":
    main()
