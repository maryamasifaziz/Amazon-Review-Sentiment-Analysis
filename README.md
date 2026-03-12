# 🛍️ Amazon Review Sentiment Analyzer

> **Fine-tuned BERT model classifying Amazon product reviews as Positive or Negative with real-time inference via an interactive Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red)
![Accuracy](https://img.shields.io/badge/Accuracy-91%25-green)

---

## 🚀 Live Demo
🔗 [View Live App](https://amazon-review-sentiment-analysis.streamlit.app/)  
🤗 [Model on HuggingFace](https://huggingface.co/Maryam3584/amazon-review-sentiment)

---

## 🧠 Overview
Fine-tuned **bert-base-uncased** on 20,000 real Amazon product reviews for binary sentiment classification. Model is hosted on HuggingFace Hub and loaded directly into a Streamlit dashboard supporting single and bulk review analysis with confidence scores and visual insights.

---

## ✨ Features
- BERT fine-tuned on 20K real Amazon reviews
- Model hosted on HuggingFace Hub — no local storage needed
- Real-time inference with confidence scores
- Bulk analysis — paste multiple reviews at once
- Confusion matrix + ROC curve visualizations
- Deployed on Streamlit Cloud with public URL

---

## 🧰 Tech Stack
| Tool | Purpose |
|---|---|
| HuggingFace Transformers | BERT model + tokenizer |
| HuggingFace Hub | Model hosting |
| PyTorch | Model training |
| Scikit-learn | Evaluation metrics |
| Streamlit | Web dashboard |
| Google Colab (T4 GPU) | Model training |

---

## 📊 Model Performance
| Metric | Score |
|---|---|
| Accuracy | 91% |
| F1 Score | 0.91 |
| AUC-ROC | 0.96 |
| Training Data | 20,000 reviews |
| Training Platform | Google Colab T4 GPU |

---

## 📁 Project Structure
```
Amazon-Review-Sentiment-Analysis/
├── modeltrain.py     # BERT fine-tuning + push to HuggingFace
├── evaluate.py       # Metrics + confusion matrix + ROC curve
├── app.py            # Streamlit dashboard
└── requirements.txt
....
```

---

## ⚙️ Run Locally
```bash
git clone https://github.com/yourusername/amazon-review-sentiment
cd amazon-review-sentiment
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏋️ Retrain Model
```bash
# Run on Google Colab with T4 GPU for best results
python train.py
```

---

## 👤 Author
**Maryam Asif**  
🎓 FAST NUCES  
🔗 [LinkedIn](https://linkedin.com/maryamasifaziz) | [GitHub](https://github.com/maryamasifaziz) | [HuggingFace](https://huggingface.co/Maryam3584)
