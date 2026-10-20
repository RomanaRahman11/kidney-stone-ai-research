# Kidney ultrasound research demo — Streamlit

**Scope:** frozen STEP06 MobileNetV3Large *research-only* image classifier. This is **not** the full proposed Gated Fusion, Simulated Federated Learning, XAI-FL, or Grad-CAM++ research system.

**Not a medical device.** Never use this app to diagnose, treat, triage or reassure patients. Source bias has been observed, and independent patient/site validation and clinician-reviewed kidney/stone localization remain unverified. Only upload non-identifiable images with appropriate authorization.

## Deployment — no retraining required

1. Unzip this folder. Upload the files/folders inside to the **root** of a GitHub repository (`app.py`, `preprocessing.py`, `requirements.txt`, `model.keras`, `.streamlit/config.toml`). GitHub usually accepts a 26 MB model with normal upload, but very large repositories may require Git LFS.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and connect the GitHub repository.
3. Choose **Create app**, set the repository and branch (e.g. `main`), and set **Main file path** to `app.py`.
4. In **Advanced settings → Python version**, select **Python 3.11**; deploy. Initial TensorFlow dependency install may take time.
5. Upload a **non-identifiable** ultrasound test image to verify that inference completes, that the processed crop appears, and that the research-only disclaimer is visible.

If Cloud shows an import error, inspect its logs. The `.keras` model has Keras 3 serialization metadata; the pinned `keras==3.13.2` matches its saved version. We have verified archive structure, preprocessing consistency and Python syntax, **but have not run TensorFlow inference in this build environment**. Check the deployment using a separate smoke test before sharing publicly.

## Technical contract

- Input: 224×224 three-channel **float32 pixels on the 0–255 scale**. The saved model includes its own MobileNetV3 `Rescaling(scale=1/127.5, offset=-1)` layer.
- Preprocessing: grayscale → ultrasound fan mask heuristic → centered 80% square crop, mask out-of-fan region → 224×224. **No CLAHE, no external division by 255.**
- Threshold is **not** presented as a diagnosis. The score is **not** calibrated and not validated externally.
- `MODEL_SHA256.txt` records the checkpoint hash and `app.py` enforces it at startup.
- The Streamlit app does **not intentionally write uploaded patient images to disk**. Standard hosting infrastructure may still process uploaded data; do not upload any identifiable medical information.

## Local testing (optional)

```bash
python -m venv .venv
# Activate the .venv, then:
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

## GitHub/Streamlit notes

- The ZIP itself is not the GitHub repository: extract it and upload its files, then point Streamlit to `app.py`.
- Public Streamlit hosting creates a live URL only **after** you deploy through your own account.
- No external validation, clinical device approval, or deployment is implied by packaging the model.
