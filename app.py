from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Keyword sets

intensifiers = {'very', 'extremely', 'really', 'super', 'so', 'too'}
negation_words = {'not', 'no', 'never'}

# Text cleaning function

def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text).lower())
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# Load ML model and vectorizer

lr_model = joblib.load("sentiment_model_lr.pkl")
vectorizer = joblib.load("vectorizer_lr.pkl")


# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()
custom_words = {
    "dependable": 2.5,
    "reliable": 2.3,
    "durable": 2.2,
    "sturdy": 2.0,
    "trustworthy": 2.4,
    "efficient": 2.1,
    "well-built": 2.2,
    "long-lasting": 2.3,
    "high-quality": 2.4,
    "premium": 2.0,
    "excellent": 3.0,
    "worthwhile": 2.0,
    "impressive": 2.5,
    "strong": 2.0
}
analyzer.lexicon.update(custom_words)

# Flask app
app = Flask(__name__)
CORS(app)

# Sentiment prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        lowered = text.lower()

        # --- Step 1: Logistic Regression Prediction ---
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        lr_proba = lr_model.predict_proba(vec)[0]
        lr_classes = lr_model.classes_

        # --- Step 2: VADER Base Analysis ---
        vader = analyzer.polarity_scores(text)
        compound = vader["compound"]

        # --- Step 3: Negation Correction (manual overrides) ---
        if re.search(r"\bnot\s+bad\b", lowered):
            compound = 0.6  # positive
        elif re.search(r"\bnot\s+good\b", lowered):
            compound = -0.6  # negative
        elif re.search(r"\bnever\s+disappoint(s|ed|ing)?\b", lowered):
            compound = 0.7
        elif re.search(r"\bnot\s+terrible\b", lowered):
            compound = 0.6
        elif re.search(r"\bnot\s+horrible\b", lowered):
            compound = 0.6

        # --- Step 4: Intensifier Amplification ---
        if any(i in lowered.split() for i in intensifiers):
            intens_count = sum(lowered.count(i) for i in intensifiers)
            multiplier = 1.3 + (0.05 * intens_count)
            compound = max(-1.0, min(compound * multiplier, 1.0))

        # --- Step 5: Convert VADER compound to probabilities ---
        if compound >= 0.05:
            vader_proba = {"positive": compound, "neutral": 1 - compound, "negative": 0}
        elif compound <= -0.05:
            vader_proba = {"negative": abs(compound), "neutral": 1 - abs(compound), "positive": 0}
        else:
            vader_proba = {"neutral": 1.0, "positive": 0, "negative": 0}

        # --- Step 6: Combine (VADER Dominant) ---
        # alpha controls how much weight VADER gets (higher = more dominant)
        alpha = 0.85  # 85% VADER, 15% LR
        final_proba = {}
        for cls in lr_classes:
            final_proba[cls] = alpha * vader_proba.get(cls, 0.0) + \
                               (1 - alpha) * lr_proba[list(lr_classes).index(cls)]

        # --- Step 7: Normalize ---
        total = sum(final_proba.values())
        if total > 0:
            final_proba = {k: v / total for k, v in final_proba.items()}

        # --- Step 8: Determine Final Sentiment ---
        result = max(final_proba, key=final_proba.get)

        # --- Step 9: Response ---
        return jsonify({
            "sentiment": result,
            "confidence": round(float(final_proba[result]), 2),
            "probabilities": final_proba,
            "compound": round(compound, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run API
if __name__ == "__main__":
    app.run(debug=True)
