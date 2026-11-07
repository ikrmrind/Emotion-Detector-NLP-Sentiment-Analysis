# 🎭 Emotion Detector — NLP Sentiment Analysis

A beautiful, AI-powered **emotion detection** web app that analyzes text and predicts the underlying emotion using classic NLP techniques — built with **Streamlit**, **scikit-learn**, and **joblib**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

> Coming soon — or run locally in 30 seconds (see below).

---

## ✨ Features

- 🎯 **6 Emotion Classes** — joy, sadness, anger, fear, love, surprise
- 📊 **Confidence Breakdown** — see probability scores for every emotion, not just the top one
- 🎨 **Beautiful Glassmorphic UI** — dark gradient theme with smooth animations
- ⚡ **Instant Predictions** — model loaded from `.pkl` artifacts (no retraining on startup)
- 🧠 **Classic NLP Pipeline** — TF-IDF (1-2 grams) + Logistic Regression
- 📦 **Modular Artifacts** — model, vectorizer, and label map saved separately with `joblib`

---

## 🖥️ Screenshot

<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/2334adfa-8cba-408f-92bf-2da6830c8e14" />


---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| UI | Streamlit |
| ML Model | Logistic Regression (`scikit-learn`) |
| Text Features | TF-IDF Vectorizer (1–2 n-grams) |
| Preprocessing | NLTK stopwords · regex · string ops |
| Persistence | `joblib` |
| Language | Python 3.9+ |

---

## 📂 Project Structure

```
emotion-detector/
├── app.py                     # Streamlit app
├── emotion_model.pkl          # Trained LogisticRegression model
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
├── label_map.pkl              # {int: 'emotion'} mapping
├── feature_columns.pkl        # Training column schema
├── stopwords.pkl              # Stopwords used during training
├── requirements.txt
├── README.md
└── notebooks/
    └── NLP_Program.ipynb      # Training notebook
```

---

## ⚙️ Installation & Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/emotion-detector.git
cd emotion-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure the model artifacts exist

You should have these files in the root folder:

```
emotion_model.pkl
tfidf_vectorizer.pkl
label_map.pkl
feature_columns.pkl
stopwords.pkl
```

> 💡 If they're missing, run the training notebook first (`notebooks/NLP_Program.ipynb`) — it saves all artifacts using `joblib`.

### 4. Launch the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## 🧪 How It Works

1. **Input** — User types a sentence into the text box.
2. **Preprocessing** — Lowercase → remove punctuation → strip digits → drop URLs → remove non-ASCII → filter stopwords.
3. **Vectorization** — Clean text is converted using the saved **TF-IDF vectorizer**.
4. **Prediction** — The **Logistic Regression** model outputs probabilities for all 6 emotions.
5. **Rendering** — Top emotion is displayed in a colorful card, plus a full breakdown bar chart.

---

## 📊 Model Performance

Trained on **16,000+** emotion-labeled samples.

| Metric | Score |
|--------|-------|
| Accuracy | **~85%** |
| Macro F1 | ~0.79 |
| Weighted F1 | ~0.85 |

**Class distribution:**

```
joy ........ 5361
sadness .... 4666
anger ...... 2159
fear ....... 1937
love ....... 1304
surprise .... 572
```

---

## 📦 Saving Artifacts (for reproducibility)

Inside the training notebook:

```python
import joblib

joblib.dump(model, "emotion_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
joblib.dump({v: k for k, v in label_map.items()}, "label_map.pkl")
joblib.dump(list(df.columns), "feature_columns.pkl")
joblib.dump(list(STOP_WORDS), "stopwords.pkl")
```

> ⚠️ The vectorizer **must** be the exact same object used to train the model — otherwise predictions will be garbage.

---

## 🎨 Emotion Palette

| Emotion | Emoji | Color |
|---------|-------|-------|
| Joy | 😄 | `#FFD93D` |
| Sadness | 😢 | `#60A5FA` |
| Anger | 😠 | `#F87171` |
| Fear | 😨 | `#A78BFA` |
| Love | ❤️ | `#F472B6` |
| Surprise | 😲 | `#FB923C` |

---

## 🧩 Requirements

```txt
streamlit>=1.30
scikit-learn>=1.3
joblib>=1.3
numpy>=1.24
nltk>=3.8
pandas>=2.0
```

Save as `requirements.txt` and install with `pip install -r requirements.txt`.

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you'd like to change.

---

## 🙏 Acknowledgements

- Dataset inspired by the **Emotion Dataset** (Saravia et al., 2018)
- Built with ❤️ using Streamlit & scikit-learn

---

## ⭐ Show Your Support

If this project helped you, give it a ⭐ on GitHub — it means a lot!

```
⭐ Star this repo if you found it useful!
```

---

**Made with 💜 by Muhammad Ikram**
```
