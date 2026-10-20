"""Local/Cloud terminal smoke test (loads model but does NOT require raw patient data)."""
from pathlib import Path
import hashlib
import json
import io
import zipfile
import numpy as np
from preprocessing import detect_roi, preprocess_for_prediction, model_batch

ROOT = Path(__file__).resolve().parent
EXPECTED = (ROOT / 'MODEL_SHA256.txt').read_text().strip().split()[0]
actual = hashlib.sha256((ROOT / 'model.keras').read_bytes()).hexdigest()
assert actual == EXPECTED, "Model hash mismatch"
with zipfile.ZipFile(ROOT / 'model.keras') as z:
    assert z.testzip() is None, "Corrupt model.keras"
    cfg = json.loads(z.read('config.json'))
    assert cfg['class_name'] == 'Functional'

synthetic = np.zeros((224, 224), dtype=np.uint8)
synthetic[30:190, 50:160] = 125
mask, status = detect_roi(synthetic)
prepped, stat2 = preprocess_for_prediction(synthetic)
x = model_batch(prepped)
assert x.shape == (1, 224, 224, 3)
assert x.dtype == np.float32
assert x.max() > 1, "Input scale should be 0–255, not 0–1"
print('Preprocessing and model archive validation: PASS')

try:
    import tensorflow as tf
except ImportError:
    print('TensorFlow is not installed: model loading/inference NOT verified')
    raise SystemExit(2)
model = tf.keras.models.load_model(str(ROOT / 'model.keras'), compile=False, safe_mode=True)
pred = np.asarray(model(x, training=False)).reshape(-1)
assert pred.shape == (1,)
assert np.isfinite(pred).all() and ((0 <= pred) & (pred <= 1)).all()
print('TensorFlow model loading and synthetic inference: PASS')
