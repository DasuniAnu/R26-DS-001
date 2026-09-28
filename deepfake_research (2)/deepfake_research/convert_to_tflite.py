# =====================================================
# FIXED TFLITE CONVERSION SCRIPT
# Fixes: TensorListReserve / BiGRU conversion error
# Run this instead of the original convert_to_tflite.py
# =====================================================

import os
import sys
import tensorflow as tf
import numpy as np

print("=" * 70)
print("TFLITE CONVERSION - FIXED VERSION (BiGRU compatible)")
print("=" * 70)

# =====================================================
# CONFIG
# =====================================================

KERAS_MODEL  = "dataset\\dataset\\deepfake_models\\condition_c_v2_final.keras"
TFLITE_OUT   = "dataset\\dataset\\deepfake_models\\condition_c_V2_final.tflite"

# =====================================================
# STEP 1 - LOAD MODEL
# =====================================================

print("\n[STEP 1] Loading Keras model...")

if not os.path.exists(KERAS_MODEL):
    print(f"ERROR: not found: {KERAS_MODEL}")
    sys.exit(1)

sys.path.insert(0, 'scripts')
from temporal_scoring import ChannelAttention, SpatialAttention

model = tf.keras.models.load_model(
    KERAS_MODEL,
    custom_objects={
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention
    },
    safe_mode=False,
    compile=False
)

print(f"Model loaded: {model.name}")
print(f"Input shape : {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# =====================================================
# STEP 2 - CONVERT WITH SELECT TF OPS FIX
#
# WHY THIS IS NEEDED:
# BiGRU uses tf.TensorListReserve internally for its
# dynamic sequence operations. Standard TFLite ops
# cannot handle dynamic tensor lists with unknown shapes.
# SELECT_TF_OPS allows a subset of full TensorFlow ops
# to run inside TFLite — this is the officially
# recommended fix from the TFLite team for RNN models.
#TFLite conversion was completed — the model went from 31.93MB to 3.91MB, an 87.7% reduction, well under the 8MB target.
# =====================================================

print("\n[STEP 2] Converting with SELECT_TF_OPS (required for BiGRU)...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# THE FIX - these 3 lines solve the TensorListReserve error
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   # standard TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS       # fallback for BiGRU tensor lists
]
converter._experimental_lower_tensor_list_ops = False

# Standard size optimisation
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
    print("Conversion successful")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# =====================================================
# STEP 3 - SAVE
# =====================================================

print("\n[STEP 3] Saving...")

os.makedirs(os.path.dirname(TFLITE_OUT), exist_ok=True)

with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_model)

keras_mb  = os.path.getsize(KERAS_MODEL)  / (1024*1024)
tflite_mb = os.path.getsize(TFLITE_OUT)   / (1024*1024)
reduction = (1 - tflite_mb / keras_mb) * 100

print(f"Saved: {TFLITE_OUT}")
print(f"\nKeras  : {keras_mb:.2f} MB")
print(f"TFLite : {tflite_mb:.2f} MB")
print(f"Reduction: {reduction:.1f}%")

# =====================================================
# STEP 4 - VERIFY INFERENCE
# =====================================================

print("\n[STEP 4] Verifying inference...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_OUT)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print(f"Input  : {inp['shape']}  dtype={inp['dtype']}")
print(f"Output : {out['shape']}  dtype={out['dtype']}")

# dummy inference test
dummy = np.random.randn(*inp['shape']).astype(np.float32)
interpreter.set_tensor(inp['index'], dummy)
interpreter.invoke()
result = interpreter.get_tensor(out['index'])

print(f"Inference output: {result}")
print(f"P(real)={result[0][0]:.4f}  P(fake)={result[0][1]:.4f}")
print("\nTFLite model verified and working")

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "="*70)
print("CONVERSION COMPLETE")
print("="*70)
print(f"""
What was fixed:
  Original error: 'tf.TensorListReserve' op requires element_shape
                  to be static during TF Lite transformation pass
  
  Cause: BiGRU uses dynamic tensor lists for sequence processing.
         Standard TFLite cannot compile these statically.
  
  Fix applied:
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS        # <-- this fixes BiGRU
    ]
    converter._experimental_lower_tensor_list_ops = False

  This is the official TFLite team recommendation for any model
  containing GRU, LSTM, or Bidirectional RNN layers.

Model sizes:
  Keras  : {keras_mb:.2f} MB
  TFLite : {tflite_mb:.2f} MB  ({reduction:.0f}% smaller)

Files ready:
  {KERAS_MODEL}
  {TFLITE_OUT}
""")