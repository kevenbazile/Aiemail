from flask import Flask, request, jsonify
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load AI Model for Smart Replies
reply_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")  # Small & free GPT model

# ✅ Load Trained Spam Detection Model
models_dir = os.path.join(os.getcwd(), 'models')
spam_model_path = os.path.join(models_dir, 'spam_classifier.pkl')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

with open(spam_model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# ✅ 1️⃣ Spam Detection API
@app.route('/predict', methods=['POST'])
def predict():
    """ Predict if an email is Spam or Not Spam """
    data = request.json
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "No email provided"}), 400

    email_transformed = vectorizer.transform([email_text])
    prediction = model.predict(email_transformed)[0]
    label = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"email": email_text, "prediction": label})

# ✅ 2️⃣ Smart Replies API
@app.route('/smart_reply', methods=['POST'])
def smart_reply():
    """ Generate 3 Smart Reply Suggestions for an Email """
    data = request.json
    email_text = data.get("email", "")

    if not email_text:
        return jsonify({"error": "No email provided"}), 400

    # ✅ Generate 3 Smart Replies
    responses = reply_generator(f"Generate a short email response to: {email_text}", max_length=50, num_return_sequences=3)
    
    reply_suggestions = [resp['generated_text'].strip() for resp in responses]

    return jsonify({
        "email": email_text,
        "smart_replies": reply_suggestions
    })

# ✅ Run Flask App
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
