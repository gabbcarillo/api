# ---------------- Imports ----------------
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, re, joblib, json, threading, random
import pandas as pd
from datetime import datetime
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, opinion_lexicon
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from wordcloud import WordCloud
from langdetect import detect, LangDetectException
import nltk

# ---------------- Flask Setup ----------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
DASHBOARD_FOLDER = "dashboard_data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

# ---------------- NLP Setup ----------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('opinion_lexicon')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

# Custom lexicon (from your real-time model)
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

# ---------------- Load Model ----------------
lr_model = joblib.load("sentiment_model_lr.pkl")
vectorizer = joblib.load("vectorizer_lr.pkl")

intensifiers = {'very', 'extremely', 'really', 'super', 'so', 'too'}

# ---------------- Shared Functions ----------------
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def is_english(text):
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def count_sentences(text):
    if not text or not isinstance(text, str):
        return 0
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if s])

# ---------------- Real-Time Prediction ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        lr_proba = lr_model.predict_proba(vec)[0]
        lr_classes = lr_model.classes_

        vader = analyzer.polarity_scores(text)
        compound = vader["compound"]

        if any(i in text.lower().split() for i in intensifiers):
            intens_count = sum(text.lower().count(i) for i in intensifiers)
            multiplier = 1.3 + (0.05 * intens_count)
            compound = max(-1.0, min(compound * multiplier, 1.0))

        if compound >= 0.05:
            vader_proba = {"positive": compound, "neutral": 1 - compound, "negative": 0}
        elif compound <= -0.05:
            vader_proba = {"negative": abs(compound), "neutral": 1 - abs(compound), "positive": 0}
        else:
            vader_proba = {"neutral": 1.0, "positive": 0, "negative": 0}

        alpha = 0.85
        final_proba = {}
        for cls in lr_classes:
            final_proba[cls] = alpha * vader_proba.get(cls, 0.0) + (1 - alpha) * lr_proba[list(lr_classes).index(cls)]

        total = sum(final_proba.values())
        if total > 0:
            final_proba = {k: v / total for k, v in final_proba.items()}

        result = max(final_proba, key=final_proba.get)

        return jsonify({
            "sentiment": result,
            "confidence": round(float(final_proba[result]), 2),
            "probabilities": final_proba,
            "compound": round(compound, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Upload + Auto-Update ----------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]
    product_name = request.form.get("product_name", "").strip().replace(" ", "_")
    if not product_name:
        return jsonify({"status": "error", "message": "Product name missing"}), 400
    if not file.filename.endswith(".xlsx"):
        return jsonify({"status": "error", "message": "Only .xlsx files allowed"}), 400

    folder_path = os.path.join(UPLOAD_FOLDER, product_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file.filename)
    file.save(file_path)

    # Process all files
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".xlsx")]
    merged_df_list = []

    for fpath in all_files:
        try:
            sheets = pd.read_excel(fpath, sheet_name=None)
            for _, df in sheets.items():
                if "review_text" in df.columns:
                    df = df[df["review_text"].notna() & df["review_text"].str.strip().ne("")]
                    df["cleaned_text"] = df["review_text"].apply(clean_text)
                    df["sentiment"] = df["cleaned_text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
                    df["sentiment"] = df["sentiment"].apply(lambda s: "positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral")
                    merged_df_list.append(df[["review_text", "cleaned_text", "sentiment"]])
        except Exception as e:
            print(f"Skipping file {fpath}: {e}")

    if not merged_df_list:
        return jsonify({"status": "error", "message": "No valid reviews found"}), 400

    merged_df = pd.concat(merged_df_list, ignore_index=True).drop_duplicates(subset=["review_text"])
    final_path = os.path.join(folder_path, f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        merged_df.to_excel(writer, index=False)

    threading.Thread(target=build_dashboard, args=(product_name, merged_df)).start()
    return jsonify({"status": "success", "message": f"{product_name} uploaded and processed."})

# ---------------- Dashboard Builder ----------------
def build_dashboard(product_name, merged_df):
    counts = merged_df["sentiment"].value_counts(normalize=True) * 100
    pos, neg, neu = round(counts.get("positive", 0)), round(counts.get("negative", 0)), round(counts.get("neutral", 0))
    avg_rating = round(5 * ((pos + 0.5 * neu) / 100), 2)

    adj_counter = Counter()
    for text in merged_df["cleaned_text"]:
        tagged = pos_tag(word_tokenize(text))
        adj_counter.update([w for w, t in tagged if t.startswith("JJ")])

    wc = WordCloud(background_color="white", width=600, height=400)
    wc.generate_from_frequencies(adj_counter)
    wc_file = os.path.join(DASHBOARD_FOLDER, f"{product_name}_wc.png")
    wc.to_file(wc_file)

    positive_reviews = merged_df[merged_df["sentiment"] == "positive"]["review_text"].tolist()
    english_reviews = [r for r in positive_reviews if is_english(r)]
    short_reviews = [r for r in english_reviews if count_sentences(r) <= 4]
    sample_reviews = random.sample(short_reviews, min(3, len(short_reviews))) if short_reviews else []

    summary = {
        "product_name": product_name.replace("_", " "),
        "avg_rating": avg_rating,
        "sentiment_breakdown": {"positive": pos, "negative": neg, "neutral": neu},
        "customer_descriptions": [{"label": w, "percent": c} for w, c in adj_counter.most_common(10)],
        "sample_reviews": sample_reviews
    }

    with open(os.path.join(DASHBOARD_FOLDER, f"{product_name}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)
