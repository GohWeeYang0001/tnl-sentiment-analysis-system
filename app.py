import re
import joblib
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# Load NLP resources
# ============================================================

@st.cache_resource
def load_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


stop_words, lemmatizer = load_nltk_resources()


# ============================================================
# Load trained model and TF-IDF vectorizer
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("svm_linearsvc_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# Text preprocessing
# ============================================================

def preprocess_text(text):
    text = str(text).lower()

    # Remove URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # Remove non-English characters, numbers, and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization, stopword removal, and lemmatization
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)


def predict_sentiment(comment):
    cleaned_text = preprocess_text(comment)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction, cleaned_text


# ============================================================
# Streamlit Page
# ============================================================

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Social Media and Public Opinion Sentiment Analysis System")

st.write(
    "This web application analyses Malaysian social media comments and predicts whether the sentiment is Positive, Neutral, or Negative."
)

st.info(
    "The deployed model is SVM / LinearSVC with TF-IDF feature extraction. "
    "RoBERTa/BERT-based model was evaluated as an advanced transformer model during model comparison."
)

st.divider()


# ============================================================
# Single Comment Prediction
# ============================================================

st.header("Single Comment Sentiment Prediction")

user_comment = st.text_area(
    "Enter a social media comment:",
    placeholder="Example: This Raya promotion is not worth it, the discount is fake."
)

if st.button("Predict Sentiment"):
    if user_comment.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        prediction, cleaned_text = predict_sentiment(user_comment)

        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success(f"Sentiment: {prediction}")
        elif prediction == "Negative":
            st.error(f"Sentiment: {prediction}")
        else:
            st.info(f"Sentiment: {prediction}")

        with st.expander("Show pre-processed text"):
            st.code(cleaned_text)


st.divider()


# ============================================================
# Batch CSV Prediction
# ============================================================

st.header("Batch CSV Sentiment Analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing social media comments",
    type=["csv"]
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(batch_df.head())

    columns = list(batch_df.columns)

    if "comment" in columns:
        default_index = columns.index("comment")
    elif "message" in columns:
        default_index = columns.index("message")
    elif "translated_message" in columns:
        default_index = columns.index("translated_message")
    else:
        default_index = 0

    text_column = st.selectbox(
        "Select the column that contains comments:",
        columns,
        index=default_index
    )

    if st.button("Analyse CSV"):
        # Predict sentiment for all selected comments
        batch_df["cleaned_text"] = batch_df[text_column].apply(preprocess_text)
        X_batch = vectorizer.transform(batch_df["cleaned_text"])
        batch_df["predicted_sentiment"] = model.predict(X_batch)

        st.subheader("Prediction Results")
        st.dataframe(batch_df)

        # ====================================================
        # Public Opinion Summary
        # ====================================================

        st.subheader("Public Opinion Summary")

        sentiment_order = ["Positive", "Neutral", "Negative"]
        total_comments = len(batch_df)

        sentiment_counts = (
            batch_df["predicted_sentiment"]
            .value_counts()
            .reindex(sentiment_order, fill_value=0)
        )

        sentiment_percentages = (sentiment_counts / total_comments * 100).round(2)

        summary_df = pd.DataFrame({
            "Sentiment": sentiment_order,
            "Count": sentiment_counts.values,
            "Percentage (%)": sentiment_percentages.values
        })

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Comments", total_comments)
        col2.metric("Positive", f"{sentiment_counts['Positive']} ({sentiment_percentages['Positive']}%)")
        col3.metric("Neutral", f"{sentiment_counts['Neutral']} ({sentiment_percentages['Neutral']}%)")
        col4.metric("Negative", f"{sentiment_counts['Negative']} ({sentiment_percentages['Negative']}%)")

        st.write("Summary table:")
        st.dataframe(summary_df)

        max_count = sentiment_counts.max()
        dominant_sentiments = sentiment_counts[sentiment_counts == max_count].index.tolist()

        if len(dominant_sentiments) == 1:
            dominant_sentiment = dominant_sentiments[0]

            if dominant_sentiment == "Positive":
                st.success(f"Dominant Public Opinion: {dominant_sentiment}")
                st.write(
                    "Overall, the analysed comments show a more positive public opinion."
                )
            elif dominant_sentiment == "Negative":
                st.error(f"Dominant Public Opinion: {dominant_sentiment}")
                st.write(
                    "Overall, the analysed comments show a more negative public opinion."
                )
            else:
                st.info(f"Dominant Public Opinion: {dominant_sentiment}")
                st.write(
                    "Overall, the analysed comments are mostly neutral or informational."
                )
        else:
            st.warning(
                "Dominant Public Opinion: Tie between "
                + ", ".join(dominant_sentiments)
            )
            st.write(
                "The analysed comments show a mixed public opinion because two or more sentiment classes have the same highest count."
            )

        # ====================================================
        # Sentiment Distribution Chart
        # ====================================================

        st.subheader("Sentiment Distribution")

        chart_df = summary_df.set_index("Sentiment")[["Count"]]
        st.bar_chart(chart_df)

        # ====================================================
        # Optional: Compare with expected_sentiment if available
        # ====================================================

        if "expected_sentiment" in batch_df.columns:
            st.subheader("Optional Evaluation on Uploaded Sample")

            correct_predictions = (
                batch_df["expected_sentiment"] == batch_df["predicted_sentiment"]
            ).sum()

            sample_accuracy = correct_predictions / total_comments * 100

            st.write(
                f"Sample Accuracy based on `expected_sentiment`: "
                f"**{sample_accuracy:.2f}%** "
                f"({correct_predictions}/{total_comments} correct)"
            )

            st.caption(
                "This accuracy is only for the uploaded sample file when an expected_sentiment column is provided. "
                "The official model evaluation is reported separately using the test set."
            )

        # ====================================================
        # Download Results
        # ====================================================

        csv_output = batch_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_output,
            file_name="sentiment_prediction_results.csv",
            mime="text/csv"
        )


st.divider()

st.caption("TNL6323 Natural Language Processing Group Project")