import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# ==== Load model & vectorizer ====
lr_model = joblib.load('./Notebook/logreg_model.pkl')
tfidf_vectorizer = joblib.load('./Notebook/vectorizer_model.pkl')

stop_words = set(stopwords.words('english'))

# ==== Text cleaning function ====
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# ==== Prediction function ====
def predict_sentiment(review_text):
    cleaned_review = clean_text(review_text)
    vec = tfidf_vectorizer.transform([cleaned_review])
    pred = lr_model.predict(vec)[0]
    prob = lr_model.predict_proba(vec)[0]
    
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    
    return sentiment, confidence

# ==== Streamlit UI ====
st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="centered")
st.title("Reddit Comment Sentiment Analysis")
st.markdown("Analyze your Reddit comments and find out if they are **positive** or **negative**!")

st.markdown("---")
review_text = st.text_area("Enter your comment here:", height=150)

# Predict button
if st.button("Predict Sentiment"):
    if review_text.strip():
        with st.spinner("Analyzing comment..."):
            sentiment, confidence = predict_sentiment(review_text)
        
        # Display sentiment
        if sentiment == "Positive":
            st.success(f"**Predicted Sentiment:** {sentiment}")
        else:
            st.error(f"**Predicted Sentiment:** {sentiment}")
        
        # Display confidence
        st.write(f"**Confidence:** {confidence:.2%}")
        st.progress(int(confidence * 100))
    else:
        st.warning("Please enter a comment to analyze.")