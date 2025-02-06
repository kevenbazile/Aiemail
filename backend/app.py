from flask import Flask, request, jsonify
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Add a homepage route to prevent 404 errors
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the AI Email Spam Detector API! Use the `/predict` endpoint to classify emails."})

# ✅ Load Trained Model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
spam_model_path = os.path.join(models_dir, 'spam_classifier.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

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
    app.run(host="0.0.0.0", port=10000, debug=True)
