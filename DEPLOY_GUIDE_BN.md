# Kidney Ultrasound Research Demo — Live Deployment (বাংলা)

## কী আছে

`app.py` (Streamlit), `model.keras` (STEP06 MobileNetV3Large; প্রায় 26 MB), `preprocessing.py` (Training-এর সঙ্গে মিলিয়ে নেওয়া preprocessing), `requirements.txt` এবং `.streamlit/config.toml`। এই Demo **STEP08 Gated Fusion, Federated Learning বা Grad-CAM++ নয়**।

## Live করতে ৭টি কাজ

1. `Kidney_Stone_Streamlit_DEPLOY_READY.zip` **Extract** করো। ZIP-টি নিজে GitHub-এ Upload না করে, ভেতরের ফাইলগুলো Upload করবে।
2. GitHub-এ লগইন করে **New repository** চাপো। নাম দিতে পারো `kidney-ultrasound-research-demo`। Public বা Private বেছে নাও (Research Model প্রকাশের অনুমতি নিশ্চিত করো)।
3. Repo-তে **Add file → Upload files**। `app.py`, `preprocessing.py`, `model.keras`, `requirements.txt`, `README.md`, `MODEL_SHA256.txt`, `DEPLOYMENT_STATUS.json`, `smoke_test.py` এবং `.streamlit/config.toml` রাখবে। `app.py` অবশ্যই repo-র **root**-এ থাকবে।
4. সব Upload হলে **Commit changes**। `model.keras` ফাইলটি GitHub-এ দেখা যাচ্ছে কি না নিশ্চিত করো; **Git LFS pointer text** হলে সরাসরি Model Load হবে না।
5. Streamlit Community Cloud (https://share.streamlit.io) খুলে GitHub Connect করো, **Create app** বেছে নাও। Repo + branch `main` নির্বাচন করো; **Main file path:** `app.py`।
6. **Advanced settings → Python version → 3.11** সেট করে **Deploy** চাপো। TensorFlow dependency প্রথমবার Install হতে কয়েক মিনিট বেশি লাগতে পারে।
7. সাইট চালু হলে শুধুমাত্র **পরিচয়বিহীন** Ultrasound Image দিয়ে কাজ করছে কি না পরীক্ষা করো। Public URL পরে শেয়ার করতে পারবে।

### Error হলে

- `No module named tensorflow` — `requirements.txt` repo-root-এ আছে কি না দেখো; Streamlit App Logs দেখো।
- `Could not load model` — `model.keras` সত্যিই আপলোড হয়েছে কি না, গিট ফাইলে LFS pointer হয়েছে কি না দেখো।
- `Model hash mismatch` — কোড/মডেল আলাদা Version হলে একই ZIP থেকে পুনরায় Upload করো।
- `Resource limit` — free Cloud-এর Memory কম হলে Paid CPU hosting বা অন্য supported service লাগতে পারে।

**Not a medical device.** Model Score রোগনির্ণয় নয়। Uploaded Patient Identifiers/Personal Health Information দেওয়া যাবে না। Human expert review ও স্বাধীন Patient/Site Validation হয়নি।

## Verification সীমা

ZIP ও Keras Archive Integrity, Preprocessing Code Equivalence (synthetic images) এবং Syntax যাচাই করা হয়েছে। এই পরিবেশে TensorFlow/Streamlit Install করা নেই, ফলে **Full Inference/Live Hosting এখনো পরীক্ষা হয়নি**। Deploy করার পর একটি বাস্তব Inference Smoke Test আবশ্যক।
