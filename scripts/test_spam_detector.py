import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Get Absolute Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
data_path = os.path.join(base_dir, 'data', 'email.csv')

spam_model_path = os.path.join(models_dir, 'spam_classifier.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

# ✅ Check If Files Exist
if not os.path.exists(spam_model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("❌ Model files not found! Ensure you downloaded them from Colab.")

# ✅ Load Model & Vectorizer
with open(spam_model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Using Colab-Trained Model!")

# ✅ Load Test Dataset
df = pd.read_csv(data_path, encoding='latin-1')[['text', 'label']]
df['label'] = df['label'].astype(int)

# ✅ Transform Text Data
X_test = vectorizer.transform(df['text'])
y_test = df['label']

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Compute Accuracy Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n🚀 **Benchmark Results (Using `email.csv`)**:")
print(f"📌 Accuracy: {accuracy:.4f}")
print(f"📌 Precision: {precision:.4f}")
print(f"📌 Recall: {recall:.4f}")
print(f"📌 F1 Score: {f1:.4f}")

# ✅ Test on Specific Emails
test_emails = [
    "Win a free iPhone!",
    "Can we reschedule our meeting?",
    "Get rich quick now!",
    "Earn $10,000 fast with this system!",
    "Let's meet for coffee tomorrow."
]

# ✅ Transform & Predict
test_emails_transformed = vectorizer.transform(test_emails)
test_predictions = model.predict(test_emails_transformed)

# ✅ Display Results
print("\n📌 Spam Detection Results:")
for email, prediction in zip(test_emails, test_predictions):
    label = "Spam" if prediction == 1 else "Not Spam"
    print(f"📧 Email: '{email}' → {label}")
