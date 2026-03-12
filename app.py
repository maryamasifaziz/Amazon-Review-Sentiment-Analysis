import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Amazon Review Sentiment Analyzer")
st.markdown("*Powered by BERT — Fine-tuned on 20,000 Amazon Reviews*")

# Load Model 
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('Maryam3584/amazon-review-sentiment')
    model = BertForSequenceClassification.from_pretrained('Maryam3584/amazon-review-sentiment')
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Predict Function 
def predict_sentiment(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = model(**encoding)
    probs = torch.softmax(output.logits, dim=1).numpy()[0]
    label = "Positive ✅" if np.argmax(probs) == 1 else "Negative ❌"
    confidence = max(probs) * 100
    return label, confidence, probs

# Single Review Tab 
tab1, tab2 = st.tabs(["Single Review", "Bulk Analysis"])

with tab1:
    st.subheader("Analyze a Single Review")
    review = st.text_area("Paste your Amazon review here:", height=150,
                          placeholder="e.g. This product is amazing...")

    if st.button("Analyze Sentiment", type="primary"):
        if review.strip():
            with st.spinner("Analyzing..."):
                label, confidence, probs = predict_sentiment(review)

            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", label)
            col2.metric("Confidence", f"{confidence:.1f}%")
            col3.metric("Review Length", f"{len(review.split())} words")

            # Probability bar chart
            fig, ax = plt.subplots(figsize=(4, 1))
            ax.barh(['Negative', 'Positive'], probs,
                    color=['#ff4b4b', '#00cc44'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Sentiment Probabilities')
            st.pyplot(fig)
        else:
            st.warning("Please enter a review first!")

# Bulk Analysis Tab
with tab2:
    st.subheader("Analyze Multiple Reviews")
    bulk_input = st.text_area(
        "Enter multiple reviews (one per line):",
        height=200,
        placeholder="Review 1\nReview 2\nReview 3"
    )

    if st.button("Analyze All", type="primary"):
        if bulk_input.strip():
            reviews = [r.strip() for r in bulk_input.split('\n') if r.strip()]
            results = []

            with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                for r in reviews:
                    label, confidence, _ = predict_sentiment(r)
                    results.append({
                        'Review': r[:50] + '...' if len(r) > 50 else r,
                        'Sentiment': label,
                        'Confidence': f"{confidence:.1f}%"
                    })

            import pandas as pd
            results_df = pd.DataFrame(results)
            results_df = results_df.astype(str)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary
            pos = sum(1 for r in results if 'Positive' in r['Sentiment'])
            neg = len(results) - pos
            col1, col2 = st.columns(2)
            col1.metric("Positive Reviews", pos)

            col2.metric("Negative Reviews", neg)

