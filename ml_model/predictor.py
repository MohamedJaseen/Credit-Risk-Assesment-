"""
ml_model/predictor.py
======================
CNN, LSTM, TabTransformer definitions + ensemble predictor.
Models are loaded lazily (only once) and cached in memory.
"""

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

N_FEATURES = 13

# Model weights for ensemble
WEIGHTS = {"cnn": 0.30, "lstm": 0.35, "tab": 0.35}

_cache = {}   # lazy-loaded model cache


# ─── Model builders ──────────────────────────────────────────────────────────

def build_cnn(n=N_FEATURES):
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input(shape=(n, 1))
    x   = layers.Conv1D(64, 3, activation="relu", padding="same")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m   = keras.Model(inp, out, name="CNN")
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m


def build_lstm(n=N_FEATURES):
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input(shape=(n, 1))
    x   = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x   = layers.Bidirectional(layers.LSTM(32))(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m   = keras.Model(inp, out, name="LSTM")
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m


def build_tab_transformer(n=N_FEATURES):
    from tensorflow import keras
    from tensorflow.keras import layers
    embed_dim = 32
    num_heads = 4
    inp = keras.Input(shape=(n,))
    x   = layers.Dense(embed_dim * n)(inp)
    x   = layers.Reshape((n, embed_dim))(x)
    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x    = layers.LayerNormalization()(x + attn)
        ff   = layers.Dense(64, activation="gelu")(x)
        ff   = layers.Dense(embed_dim)(ff)
        x    = layers.LayerNormalization()(x + ff)
    x   = layers.Flatten()(x)
    x   = layers.Dense(64, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m   = keras.Model(inp, out, name="TabTransformer")
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m


# ─── Inference ───────────────────────────────────────────────────────────────

def _load():
    """Load all available saved models into cache."""
    from tensorflow import keras
    paths = {
        "cnn":  os.path.join(MODEL_DIR, "cnn_model.keras"),
        "lstm": os.path.join(MODEL_DIR, "lstm_model.keras"),
        "tab":  os.path.join(MODEL_DIR, "tab_model.keras"),
    }
    for key, path in paths.items():
        if key not in _cache and os.path.exists(path):
            _cache[key] = keras.models.load_model(path)


def predict_all(scaled: np.ndarray) -> dict:
    """
    Run all available models and return individual + ensemble probability.

    Returns dict: {cnn, lstm, tab, ensemble} — only keys for loaded models.
    """
    _load()

    x3d   = scaled.reshape(1, N_FEATURES, 1).astype(np.float32)
    x_flat = scaled.reshape(1, N_FEATURES).astype(np.float32)
    results = {}

    if "cnn"  in _cache: results["cnn"]  = float(_cache["cnn"].predict(x3d,    verbose=0)[0][0])
    if "lstm" in _cache: results["lstm"] = float(_cache["lstm"].predict(x3d,    verbose=0)[0][0])
    if "tab"  in _cache: results["tab"]  = float(_cache["tab"].predict(x_flat,  verbose=0)[0][0])

    if not results:
        # Fallback: rule-based estimate if no models are trained yet
        results["ensemble"] = _rule_fallback(scaled)
        return results

    total_w  = sum(WEIGHTS[k] for k in results)
    ensemble = sum(v * WEIGHTS[k] for k, v in results.items()) / total_w
    results["ensemble"] = round(ensemble, 4)
    return results


def _rule_fallback(f: np.ndarray) -> float:
    """Simple heuristic before any model is trained."""
    # Indexes: 5=missed, 6=util, 9=total_dti, 11=emp
    score = min(0.99, f[5]*0.18 + f[6]/100*0.22 + f[9]*0.30 + f[11]*0.05)
    return round(float(score), 4)
