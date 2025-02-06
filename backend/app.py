import os
from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Adjust Path for `models/` Since `app.py` is Inside `backend/`
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up one level from `backend/`
models_dir = os.path.join(base_dir, 'models')  # Corrects the path

spam_model_path = os.path.join(models_dir, 'spam_classifier.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

# ✅ Check If Model Files Exist
if not os.path.exists(spam_model_path):
    raise FileNotFoundError(f"❌ Model file NOT found: {spam_model_path}")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"❌ Vectorizer file NOT found: {vectorizer_path}")

# ✅ Load Trained Model
with open(spam_model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """ Predict if an email is Spam or Not Spam """
    data = request.json
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "No email provided"}), 400

    # ✅ Transform Input Email
    email_transformed = vectorizer.transform([email_text])

    # ✅ Make Prediction
    prediction = model.predict(email_transformed)[0]
    label = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"email": email_text, "prediction": label})

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
