import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Loading Saved Model
print("Loading saved model...")
tokenizer = BertTokenizer.from_pretrained('./models/bert-sentiment')
model = BertForSequenceClassification.from_pretrained('./models/bert-sentiment')
model.eval()

# Predict Function 
def predict(texts):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**encodings)
    probs = torch.softmax(outputs.logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs

# Sample Test Reviews 
test_reviews = [
    "This product is absolutely amazing, best purchase ever!",
    "Terrible quality, broke after one day. Complete waste of money.",
    "Decent product, does what it says nothing more nothing less.",
    "I love this so much, would definitely recommend to everyone!",
    "Worst experience ever, do not buy this product."
]
test_labels = [1, 0, 1, 1, 0]

preds, probs = predict(test_reviews)

# Classification Report
print("\nClassification Report:")
print(classification_report(
    test_labels, preds,
    target_names=['Negative', 'Positive']
))
print(f"AUC Score: {roc_auc_score(test_labels, probs[:, 1]):.4f}")

# Confusion Matrix 
cm = confusion_matrix(test_labels, preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 6. ROC Curve 
fpr, tpr, _ = roc_curve(test_labels, probs[:, 1])
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_score(test_labels, probs[:, 1]):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()