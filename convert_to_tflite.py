import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
N_FEATURES = 13

print("Loading models...")
cnn = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
lstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
tab = tf.keras.models.load_model(os.path.join(MODEL_DIR, "tab_model.keras"))

# Dummy inputs
dummy_3d = np.random.randn(1, N_FEATURES, 1).astype(np.float32)
dummy_2d = np.random.randn(1, N_FEATURES).astype(np.float32)

print("\n--- Converting CNN ---")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
    cnn_tflite = converter.convert()
    cnn_path = os.path.join(MODEL_DIR, "cnn_model.tflite")
    with open(cnn_path, "wb") as f:
        f.write(cnn_tflite)
    print(f"CNN converted successfully! Saved to {cnn_path}")
    
    # Test loading
    interpreter = tf.lite.Interpreter(model_path=cnn_path)
    interpreter.allocate_tensors()
    print("CNN interpreter loaded successfully!")
except Exception as e:
    print(f"CNN conversion failed: {e}")

print("\n--- Converting LSTM ---")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(lstm)
    # Enable select TF ops just in case LSTM needs them
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter._experimental_lower_tensor_list_ops = False
    lstm_tflite = converter.convert()
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.tflite")
    with open(lstm_path, "wb") as f:
        f.write(lstm_tflite)
    print(f"LSTM converted successfully! Saved to {lstm_path}")
    
    # Test loading
    interpreter = tf.lite.Interpreter(model_path=lstm_path)
    interpreter.allocate_tensors()
    print("LSTM interpreter loaded successfully!")
except Exception as e:
    print(f"LSTM conversion failed: {e}")

print("\n--- Converting TabTransformer ---")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(tab)
    tab_tflite = converter.convert()
    tab_path = os.path.join(MODEL_DIR, "tab_model.tflite")
    with open(tab_path, "wb") as f:
        f.write(tab_tflite)
    print(f"TabTransformer converted successfully! Saved to {tab_path}")
    
    # Test loading
    interpreter = tf.lite.Interpreter(model_path=tab_path)
    interpreter.allocate_tensors()
    print("TabTransformer interpreter loaded successfully!")
except Exception as e:
    print(f"TabTransformer conversion failed: {e}")
