from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# 1. Negation + Intensifier Cleaning
# ---------------------------
negation_words = {'not', 'no', 'never'}
intensifiers = {'very', 'extremely', 'really', 'super', 'so', 'too'}

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()

    new_tokens = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word in negation_words and i + 1 < len(tokens):
            new_tokens.append(word + '_' + tokens[i + 1])
            i += 2
            continue
        new_tokens.append(word)
        i += 1
    return ' '.join(new_tokens)

# ---------------------------
# 2. Load LR model + vectorizer
# ---------------------------
lr_model = joblib.load("sentiment_model_lr.pkl")
vectorizer = joblib.load("vectorizer_lr.pkl")  # trained with ngram_range=(1,2)

# ---------------------------
# 3. Initialize VADER
# ---------------------------
analyzer = SentimentIntensityAnalyzer()

# ---------------------------
# 4. Flask API
# ---------------------------
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        lowered_text = text.lower()

        # --- 1. LR prediction ---
        cleaned_text = clean_text(text)
        text_vec = vectorizer.transform([cleaned_text])
        lr_proba = lr_model.predict_proba(text_vec)[0]
        lr_classes = lr_model.classes_  # ['negative', 'neutral', 'positive']

        # --- 2. Improved VADER sentiment with intensifier boosting ---
        vader_scores = analyzer.polarity_scores(text)

        # Manually amplify when intensifiers present
        intensity_multiplier = 1.0
        if any(i in text.lower().split() for i in intensifiers):
            intens_count = sum(text.lower().count(i) for i in intensifiers)
            intensity_multiplier = 1.3 + (0.1 * intens_count)  # 1.3x for "very", stronger if repeated

        # Apply boost based on polarity direction
        compound = vader_scores['compound']
        if compound > 0:
            compound = min(compound * intensity_multiplier, 1.0)
        elif compound < 0:
            compound = max(compound * intensity_multiplier, -1.0)

        # Recalculate adjusted scores proportionally
        if compound >= 0.05:
            vader_proba = {"positive": compound, "neutral": 1 - compound, "negative": 0}
        elif compound <= -0.05:
            vader_proba = {"negative": abs(compound), "neutral": 1 - abs(compound), "positive": 0}
        else:
            vader_proba = {"neutral": 1.0, "positive": 0, "negative": 0}

        # --- 3. Combine LR + VADER ---
        rare_words = [w for w in cleaned_text.split() if w not in vectorizer.get_feature_names_out()]
        alpha = 0.5 if rare_words else 0.6  # give more weight to VADER
        final_proba = {}
        for cls in lr_classes:
            final_proba[cls] = alpha * lr_proba[list(lr_classes).index(cls)] + (1 - alpha) * vader_proba.get(cls, 0.0)

        # --- 4. Intensifier adjustment (stronger + position-based) ---
        intensity_boost = 1.0
        if any(i in lowered_text for i in intensifiers):
            # Count how many intensifiers appear to adjust scale
            intens_count = sum(lowered_text.count(i) for i in intensifiers)
            intensity_boost = 1.1 + (0.1 * intens_count)  # "very" = +10%, "very very" = +20%

            # Stronger scaling based on positive/negative terms nearby
            if any(word in lowered_text for word in ["bad", "terrible", "awful", "poor", "horrible"]):
                final_proba["negative"] *= intensity_boost
            elif any(word in lowered_text for word in ["good", "great", "excellent", "amazing", "fantastic"]):
                final_proba["positive"] *= intensity_boost

        # --- 5. Re-normalize probabilities ---
        total = sum(final_proba.values())
        if total > 0:
            final_proba = {k: v / total for k, v in final_proba.items()}

        # --- 6. Final prediction ---
        final_pred = max(final_proba, key=final_proba.get)

        return jsonify({
            "sentiment": final_pred,
            "confidence": round(float(final_proba[final_pred]), 2),
            "probabilities": final_proba
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
