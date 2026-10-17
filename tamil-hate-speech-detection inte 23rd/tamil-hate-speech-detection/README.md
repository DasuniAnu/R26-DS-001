# Tamil Hate Speech Detection System

**Author:** Anutthara S.A.D (IT22217554)  
**Supervisor:** Dr. Lakmini Abeywardhana  
**Co-supervisor:** Mrs. Narmadha Gamage  
**Institution:** Sri Lanka Institute of Information Technology (SLIIT)  
**Degree:** B.Sc. (Hons) Information Technology — Data Science  

---

## Research Title
Multimodal Tamil Hate Speech Detection using Text and Audio Analysis

---

## Model Results

| Model | Modality | Accuracy | F1 Score |
|-------|----------|----------|----------|
| M1: TF-IDF + Logistic Regression | Text only | 79.94% | 0.8072 |
| M2: LSTM (Bidirectional) | Text only | 80.81% | 0.8034 |
| M3: mBERT fine-tuned | Text only | 83.65% | 0.8335 |
| M4: XLM-RoBERTa fine-tuned | Text only | 83.34% | 0.8354 |
| A1: All features + SVM | Audio only | 85.86% | 0.8573 |
| A2: All features + Dense NN | Audio only | 83.79% | 0.8372 |
| F1: Early Fusion (XLM-R + Audio) | Multimodal | TBD | TBD |
| F2: Late Fusion | Multimodal | TBD | TBD |

---

## Project Structure
```
tamil-hate-speech-detection/
├── api/
│   └── app.py               # Flask API
├── demo/
│   └── streamlit_app.py     # Streamlit UI
├── models/
│   ├── text_models/
│   │   └── xlmr_saved/      # Download from Google Drive
│   ├── audio_models/
│   │   ├── audio_svm.pkl
│   │   ├── audio_scaler.pkl
│   │   └── audio_dnn.pt
│   └── fusion_models/
│       ├── fusion_model_f1.pt
│       └── fusion_scaler.pkl
├── data/
│   ├── text/
│   │   ├── train_clean.csv
│   │   ├── dev_clean.csv
│   │   └── test_clean.csv
│   └── audio/
│       └── final_audio_labels.csv
├── notebooks/
│   └── research-notebook-dasuni.ipynb
├── results/
│   └── all_model_results.csv
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download models
Download model files from Google Drive and place in correct folders.
[Model Download Link — ADD YOUR GOOGLE DRIVE LINK HERE]

### Step 3 — Start Flask API
```bash
python api/app.py
```
API runs at: http://localhost:5000

### Step 4 — Start Streamlit demo
```bash
streamlit run demo/streamlit_app.py
```
Demo runs at: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Check API status |
| POST | /predict | Predict single text |
| POST | /predict_batch | Predict multiple texts |
| POST | /predict_audio | Predict audio file |

### Example
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "dei nee enna pannra idhu wrong da"}'
```

---

## Datasets
- **Text:** DravidianCodeMix Tamil Offensive Language Dataset
- **Audio:** EmoTa Tamil Emotional Speech + YouTube Tamil clips

## Citation
```
Anutthara S.A.D. (2026). Multimodal Tamil Hate Speech Detection 
using Text and Audio Analysis. Final Year Research Project, SLIIT.
```
