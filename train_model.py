"""
train_model.py
==============
End-to-end training:
  1. Generate / load synthetic dataset
  2. Preprocess + StandardScaler
  3. Train CNN, BiLSTM, TabTransformer (each 30 epochs with early stopping)
  4. 5-fold cross-validation per model
  5. Ensemble evaluation
  6. Save models, scaler, metrics.json, model_comparison.json
  7. Save evaluation plots

Run:  python train_model.py
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ml_model.predictor import build_cnn, build_lstm, build_tab_transformer, N_FEATURES

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42); tf.random.set_seed(42)

DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = [
    "age","monthly_income","loan_amount","loan_tenure","existing_emis",
    "missed_payments","credit_utilization","credit_history_years","dti",
    "total_dti","income_loan_ratio","employment_encoded","emi_new",
]


# ─── 1. Dataset generation ──────────────────────────────────────────────────

def generate_dataset(n=7000):
    print(f"Generating {n:,} synthetic samples…")
    emp  = np.random.choice([0,1,2], n, p=[0.62,0.28,0.10])
    age  = np.random.randint(21, 65, n)
    inc  = np.where(emp==0,
              np.random.normal(60000,25000,n).clip(15000,250000),
              np.where(emp==1,
                  np.random.normal(42000,22000,n).clip(10000,180000),
                  np.random.normal(9000, 5000, n).clip(0, 18000)))
    loan = np.random.uniform(50000, 2_500_000, n)
    ten  = np.random.choice([12,24,36,48,60,84,120,180], n)
    emis = np.random.uniform(0, inc*0.55, n)
    miss = np.random.choice([0,0,0,0,1,2,3,4], n, p=[0.42,0.18,0.12,0.08,0.10,0.05,0.03,0.02])
    util = np.random.uniform(3, 97, n)
    hist = np.random.uniform(0.3, 22, n)
    rate = 0.01
    emi_n = loan * rate / (1 - (1+rate)**(-ten))
    dti   = np.clip(emis/np.maximum(inc,1), 0, 2)
    tdti  = np.clip((emis+emi_n)/np.maximum(inc,1), 0, 2)
    incr  = np.clip(inc/np.maximum(loan,1), 0, 5)

    X = np.column_stack([age,inc,loan,ten,emis,miss,util,hist,dti,tdti,incr,emp,emi_n])
    risk = (0.28*np.clip(miss/4,0,1) + 0.22*np.clip(tdti/1.5,0,1)
            + 0.20*(util/100) + 0.15*np.clip(1-hist/20,0,1) + 0.15*(emp/2))
    y = (risk + np.random.normal(0,0.04,n) > 0.44).astype(int)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    path = os.path.join(DATA_DIR, "dataset.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}  |  default rate: {y.mean():.1%}")
    return df


# ─── 2. Preprocessing ───────────────────────────────────────────────────────

def preprocess(df):
    X = df.drop("label",axis=1).values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    with open(os.path.join(MODEL_DIR,"scaler.pkl"),"wb") as f:
        pickle.dump(sc, f)
    print(f"  Train: {len(Xtr):,}  |  Test: {len(Xte):,}")
    return Xtr, Xte, ytr, yte


# ─── 3. Training ────────────────────────────────────────────────────────────

def train(model, Xtr, ytr, model_type):
    print(f"\n  Training {model_type}…")
    cbs = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_auc"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
    ]
    if model_type in ("CNN","LSTM"):
        Xtr3 = Xtr.reshape(-1, N_FEATURES, 1)
        model.fit(Xtr3, ytr, epochs=40, batch_size=256,
                  validation_split=0.15, callbacks=cbs, verbose=1)
        return model
    else:
        model.fit(Xtr, ytr, epochs=40, batch_size=256,
                  validation_split=0.15, callbacks=cbs, verbose=1)
        return model


def evaluate(model, Xte, yte, model_type, name):
    if model_type in ("CNN","LSTM"):
        prob = model.predict(Xte.reshape(-1,N_FEATURES,1), verbose=0).flatten()
    else:
        prob = model.predict(Xte, verbose=0).flatten()
    pred = (prob >= 0.5).astype(int)
    cm   = confusion_matrix(yte, pred)
    tn,fp,fn,tp = cm.ravel()
    m = {
        "model":     name,
        "accuracy":  round(accuracy_score(yte, pred), 4),
        "precision": round(precision_score(yte, pred, zero_division=0), 4),
        "recall":    round(recall_score(yte, pred, zero_division=0), 4),
        "f1":        round(f1_score(yte, pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(yte, prob), 4),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }
    print(f"  {name}: Acc={m['accuracy']}  F1={m['f1']}  AUC={m['auc']}")
    return m, prob


# ─── 4. Cross-validation ─────────────────────────────────────────────────────

def cross_validate(build_fn, X, y, model_type, k=5):
    print(f"  Cross-validating {model_type} ({k}-fold)…")
    kf   = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    aucs = []
    for fold,(tr_i,va_i) in enumerate(kf.split(X,y),1):
        m = build_fn()
        Xtr,ytr = X[tr_i], y[tr_i]
        Xva,yva = X[va_i], y[va_i]
        if model_type in ("CNN","LSTM"):
            Xtr = Xtr.reshape(-1,N_FEATURES,1); Xva = Xva.reshape(-1,N_FEATURES,1)
        m.fit(Xtr, ytr, epochs=20, batch_size=256, verbose=0,
              validation_data=(Xva,yva),
              callbacks=[keras.callbacks.EarlyStopping(patience=4,monitor="val_auc",
                                                        restore_best_weights=True)])
        prob = m.predict(Xva, verbose=0).flatten()
        auc  = roc_auc_score(yva, prob)
        aucs.append(auc)
        print(f"    Fold {fold}: AUC={auc:.4f}")
    mean_auc = np.mean(aucs)
    print(f"  → Mean AUC: {mean_auc:.4f} ± {np.std(aucs):.4f}")
    return {"mean_auc": round(mean_auc,4), "std_auc": round(np.std(aucs),4), "folds": [round(a,4) for a in aucs]}


# ─── 5. Evaluation plots ─────────────────────────────────────────────────────

def save_plots(yte, probs, comparison):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Model Evaluation — Credit Risk System", fontsize=14, fontweight="bold")
    clrs = {"CNN":"#3b82f6","LSTM":"#22c55e","TabTransformer":"#f59e0b","Ensemble":"#ef4444"}

    # ROC curves
    ax = axes[0,0]
    for name, prob in probs.items():
        fpr,tpr,_ = roc_curve(yte, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte,prob):.3f})", color=clrs.get(name,"grey"))
    ax.plot([0,1],[0,1],"k--",lw=0.8); ax.set_title("ROC Curves")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Confusion matrix (ensemble)
    ax = axes[0,1]
    ens_pred = (probs["Ensemble"]>=0.5).astype(int)
    sns.heatmap(confusion_matrix(yte,ens_pred),annot=True,fmt="d",cmap="Blues",ax=ax,
                xticklabels=["Low","High"],yticklabels=["Low","High"])
    ax.set_title("Ensemble Confusion Matrix")

    # Bar comparison
    ax = axes[0,2]
    metrics = ["accuracy","precision","recall","f1","auc"]
    models  = [n for n in comparison if n in clrs]
    x = np.arange(len(metrics)); w=0.22
    for i,(m,c) in enumerate([(n,clrs[n]) for n in models]):
        ax.bar(x+i*w, [comparison[m].get(mt,0) for mt in metrics], w, label=m, color=c, alpha=0.85)
    ax.set_xticks(x+w); ax.set_xticklabels([m.title() for m in metrics])
    ax.set_ylim(0,1.1); ax.legend(fontsize=8); ax.set_title("Model Comparison"); ax.grid(alpha=0.2,axis="y")

    # Probability distribution
    ax = axes[1,0]
    ax.hist(probs["Ensemble"][yte==0],bins=40,alpha=0.6,label="Actual Low Risk",color="#22c55e")
    ax.hist(probs["Ensemble"][yte==1],bins=40,alpha=0.6,label="Actual High Risk",color="#ef4444")
    ax.set_title("Probability Distribution"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Placeholder for cross-val (can add if cv_results passed)
    axes[1,1].set_title("Cross-Validation"); axes[1,1].text(0.5,0.5,"See console output",
                                                              ha="center",va="center",transform=axes[1,1].transAxes)

    # F1 bar
    ax = axes[1,2]
    names = [n for n in comparison]; f1s = [comparison[n]["f1"] for n in names]
    ax.barh(names, f1s, color=[clrs.get(n,"grey") for n in names]); ax.set_xlim(0,1)
    ax.set_title("F1 Score Comparison"); ax.grid(alpha=0.2,axis="x")

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "evaluation_plots.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  Evaluation plots → {path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv = os.path.join(DATA_DIR,"dataset.csv")
    df  = pd.read_csv(csv) if os.path.exists(csv) else generate_dataset()
    Xtr, Xte, ytr, yte = preprocess(df)
    Xall = np.vstack([Xtr,Xte]); yall = np.concatenate([ytr,yte])

    all_probs = {}; comparison = {}

    for model_fn, mtype, key, save_name in [
        (build_cnn,             "CNN",            "cnn",  "cnn_model.h5"),
        (build_lstm,            "LSTM",           "lstm", "lstm_model.h5"),
        (build_tab_transformer, "TabTransformer", "tab",  "tab_model.h5"),
    ]:
        m = model_fn()
        m = train(m, Xtr, ytr, mtype)
        m.save(os.path.join(MODEL_DIR, save_name))
        metrics, prob = evaluate(m, Xte, yte, mtype, mtype)
        all_probs[mtype] = prob
        comparison[mtype] = metrics
        cross_validate(model_fn, Xall, yall, mtype)

    # Ensemble
    ens_prob = all_probs["CNN"]*0.30 + all_probs["LSTM"]*0.35 + all_probs["TabTransformer"]*0.35
    ens_pred = (ens_prob>=0.5).astype(int)
    cm = confusion_matrix(yte, ens_pred); tn,fp,fn,tp = cm.ravel()
    ens_m = {
        "model":"Ensemble",
        "accuracy": round(accuracy_score(yte,ens_pred),4),
        "precision":round(precision_score(yte,ens_pred,zero_division=0),4),
        "recall":   round(recall_score(yte,ens_pred,zero_division=0),4),
        "f1":       round(f1_score(yte,ens_pred,zero_division=0),4),
        "auc":      round(roc_auc_score(yte,ens_prob),4),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
    }
    all_probs["Ensemble"] = ens_prob
    comparison["Ensemble"] = ens_m
    print(f"\n  Ensemble: Acc={ens_m['accuracy']}  F1={ens_m['f1']}  AUC={ens_m['auc']}")

    with open(os.path.join(MODEL_DIR,"metrics.json"),"w") as f:
        json.dump(ens_m, f, indent=2)
    with open(os.path.join(MODEL_DIR,"model_comparison.json"),"w") as f:
        json.dump(comparison, f, indent=2)

    save_plots(yte, all_probs, comparison)

    print("\n✅ Training complete!")
    print(f"   Models saved in: {MODEL_DIR}")
    print("   Start backend: python backend/app.py")
    print("   Start frontend: streamlit run frontend/app.py")
