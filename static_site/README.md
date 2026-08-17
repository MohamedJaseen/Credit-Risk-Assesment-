# Deploying CreditIQ Static Site to Render

This directory (`static_site/`) contains the complete lightweight static web application for **CreditIQ**. It can be deployed for **100% free** on Render as a **Render Static Site**.

---

## 🚀 Option 1: Deploy via Render Blueprint (`render.yaml`) - Recommended

The project root includes an updated `render.yaml` configured with both backend and static frontend services.

1. Push your changes to **GitHub / GitLab**.
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect your repository. Render will automatically detect `credit-risk-static-ui` static site alongside your backend API!
5. Click **Apply**. Render will deploy your static site instantly with global CDN and free automatic SSL/TLS!

---

## 🌐 Option 2: Manual 1-Click Setup on Render Dashboard

If you deployed your backend separately on Render (e.g. `https://credit-risk-assesment-1.onrender.com`), you can deploy this static site manually in 2 minutes:

1. Go to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → **Static Site**.
3. Connect your repository.
4. Fill in the following settings:
   - **Name**: `credit-risk-static-ui` (or any name you prefer)
   - **Branch**: `main` (or your active branch)
   - **Build Command**: *(leave empty)*
   - **Publish Directory**: `./static_site`
5. Click **Create Static Site**.

Render will deploy your static site!

---

## ⚙️ Connecting to your Render Backend API

1. Open your newly deployed Render Static Site URL (e.g., `https://credit-risk-static-ui.onrender.com`).
2. Click the **⚙️ API URL** button in the top navigation bar.
3. Enter your deployed Flask backend URL (e.g., `https://credit-risk-assesment-1.onrender.com`).
4. Click **Save & Test Connection**. The app will immediately test the connection to your backend `/api/health` endpoint and save the setting in local storage!

---

## 💡 Key Features of the Static Site
- **Zero Server Overhead**: Hosted entirely on CDN, fast loading worldwide.
- **REST & JWT Authentication**: Registers and logs in users directly via your backend `/api/login` and `/api/register`.
- **Real-Time Machine Learning Scoring**: Submits applications to `/api/predict`, rendering credit scores, ensemble risk probabilities, SHAP explainability charts, reason codes, and recommendations.
- **Role-Based Views**: Automatically presents Admin Dashboard and approval queue for users with `admin` role.
- **Custom Backend Endpoint Switcher**: Change backend URLs dynamically without re-building or re-deploying code.
