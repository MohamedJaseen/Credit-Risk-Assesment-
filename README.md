# CreditIQ — Credit Risk Assessment System v2

### Final Year Project | Machine Learning + FinTech | Academic Level

---

## 📁 Project Structure

```
credit_risk_v2/
│
├── backend/
│   ├── app.py          ← Flask API (all routes + RBAC)
│   ├── auth.py         ← bcrypt + JWT + role decorators
│   └── database.py     ← SQLite — all DB operations
│
├── frontend/
│   └── app.py          ← Streamlit multi-page UI
│
├── ml_model/
│   ├── feature_engineering.py  ← raw inputs → 13 features
│   ├── credit_score.py         ← rule-based 300–900 score
│   ├── decision_engine.py      ← risk category + reason codes
│   ├── predictor.py            ← CNN, BiLSTM, TabTransformer, ensemble
│   ├── explain.py              ← SHAP approximation + charts
│   └── recommendations.py      ← personalised advice
│
├── reports/
│   └── pdf_report.py   ← ReportLab PDF generation
│
├── data/               ← SQLite DB + dataset.csv (auto-created)
├── models/             ← saved .h5 files + metrics JSON (auto-created)
│
├── train_model.py      ← Full training script
└── requirements.txt
```

---

## ⚡ Quick Start (3 steps)

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Train the ML models

```bash
python train_model.py
```

This will:
- Generate 7,000 synthetic loan records → `data/dataset.csv`
- Train CNN, BiLSTM, TabTransformer
- Run 5-fold cross-validation for each
- Save ensemble metrics → `models/metrics.json`
- Save model comparison → `models/model_comparison.json`
- Generate evaluation plots → `models/evaluation_plots.png`

⏱ Training takes **3–8 minutes** depending on your hardware.

### Step 3 — Run the system (two terminals)

**Terminal 1 — Backend:**
```bash
cd backend
python app.py
# Flask starts at http://localhost:5000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
# Opens at http://localhost:8501
```

---

## 🔐 Default Credentials

| Role  | Email              | Password   |
|-------|--------------------|------------|
| Admin | admin@bank.com     | admin123   |
| User  | Register via UI    | Your choice|

---

## 🗄 Database Schema

### users table
```sql
id, name, email, password_hash, role (user/admin), created_at
```

### applications table — single table with status column
```sql
id, user_id, user_name, user_email,
input_json,                    -- raw user inputs
credit_score, score_label,     -- computed credit score
probability, risk_category,    -- ML prediction
ml_recommendation,             -- "Recommend: Approve/Reject"
reason_codes, shap_explanation, recommendations,
status,                        -- Pending | Approved | Rejected ← WORKFLOW COLUMN
reviewed_by, review_note, reviewed_at,
created_at
```

### decisions table — audit trail
```sql
id, application_id, admin_email,
old_status, new_status, note, decided_at
```

---

## 🌐 API Reference

| Method | Endpoint | Access | Description |
|--------|----------|--------|-------------|
| POST | /api/register | Public | Create user account |
| POST | /api/login | Public | Get JWT token |
| POST | /api/predict | User | Submit loan application |
| GET | /api/my-applications | User | Own application history |
| GET | /api/application/<id> | User/Admin | Single application detail |
| GET | /api/admin/dashboard | Admin | Stats + monthly trend |
| GET | /api/admin/applications | Admin | All applications |
| POST | /api/admin/decision/<id> | Admin | Approve / Reject |
| GET | /api/admin/decision-history/<id> | Admin | Audit log |
| GET | /api/admin/users | Admin | All registered users |
| GET | /api/metrics | Admin | Ensemble model metrics |
| GET | /api/model-comparison | Admin | CNN vs LSTM vs TabTransformer |
| POST | /api/admin/train | Admin | Trigger retraining |
| GET | /api/health | Public | Health check |

---

## � Render Deployment

This project is set up for Render with two services:

- Backend service: exposes the Flask API at /api/*
- Frontend service: runs Streamlit and calls the backend through the API_URL environment variable

### Required Render environment variables

For the frontend service, set:

```bash
API_URL=https://<your-backend-service>.onrender.com
```

For the backend service, Render will provide PORT automatically. The app is already configured to bind to that port.

### Recommended Render setup

1. Create a web service for the backend using the repository.
2. Use the start command:
   ```bash
   gunicorn --config gunicorn.conf.py backend.app:app
   ```
3. Create a second web service for the frontend.
4. Use the start command:
   ```bash
   streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
5. Set the frontend's API_URL to the backend service URL.

---

## �🔒 RBAC Rules

```
USER role:
  ✅ Register, login
  ✅ Submit loan application
  ✅ View own applications and results
  ✅ Download PDF report
  ❌ Cannot access /api/admin/* routes
  ❌ Cannot see other users' data
  ❌ Cannot approve/reject applications
  ❌ Cannot train models

ADMIN role:
  ✅ All user features
  ✅ View all applications (with filters)
  ✅ Approve or reject applications
  ✅ View analytics dashboard
  ✅ View model performance metrics
  ✅ Compare CNN vs LSTM vs TabTransformer
  ✅ Trigger model retraining
  ✅ View user management
  ✅ Full decision audit history
```

---

## 🔄 Real-World Workflow

```
User submits application
       ↓
Status = "Pending"
       ↓
ML models run: CNN + BiLSTM + TabTransformer ensemble
       ↓
Credit score computed (300–900) — not entered by user
       ↓
Risk category: LOW / MEDIUM / HIGH
ML Recommendation: "Recommend: Approve" or "Recommend: Reject"
       ↓
Admin reviews in dashboard
       ↓
Admin clicks Approve or Reject (with optional note)
       ↓
Status updated to "Approved" / "Rejected"
Decision logged in decisions table
       ↓
User sees final status in "My Applications"
```

---

## 🤖 ML Architecture

### Three Models
| Model | Architecture | Strength |
|-------|-------------|---------|
| CNN | 1-D Conv over feature vector | Local feature interactions |
| BiLSTM | Bidirectional LSTM | Sequential feature patterns |
| TabTransformer | Multi-Head Attention MLP | Global feature interactions |

### Ensemble
```
Final Probability = CNN×0.30 + BiLSTM×0.35 + TabTransformer×0.35
```

### Credit Score Formula
```
Payment History     × 35%
Credit Utilization  × 30%
Credit History      × 15%
Income Stability    × 10%
DTI / Loan Burden   × 10%
= Score (300–900)
```

---

## 🎓 Academic Notes

- Uses **SQLite** — appropriate for college project (no server setup)
- **SHAP approximation** used for speed; swap `approx_shap()` for `compute_shap()` (full KernelExplainer) for more accurate explanations
- Synthetic dataset generated programmatically — replace with real lending data for better results
- All endpoints include proper RBAC — suitable for demonstrating access control concepts
- Decision workflow (Pending → Approved/Rejected) mirrors real fintech applications
