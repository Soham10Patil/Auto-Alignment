import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Download NLTK Sentiment Lexicon
import nltk
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to compute sentiment score
def get_sentiment_score(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

# Function to calculate recency score
def calculate_recency_score(timestamp):
    days_since_publication = (datetime.now() - datetime.strptime(timestamp, '%Y-%m-%d')).days
    return max(0, 1 - (days_since_publication / 7))

# Function to calculate priority score
def calculate_priority(row, category_weights, location_weights):
    return (0.4 * row['Recency'] +
            0.3 * row['Sentiment'] +
            0.2 * row['Engagement'] +
            0.1 * category_weights[row['Category']] * location_weights[row['Category']])

# Default category weights
CATEGORY_WEIGHTS = {"Business": 0.2, "Sports": 0.1, "Technology": 0.15, "Politics": 0.25, "Entertainment": 0.1}

# Location-specific weights
LOCATION_WEIGHTS = {
    "Mumbai": {"Business": 1.2, "Sports": 0.8, "Technology": 1.0, "Politics": 1.0, "Entertainment": 1.1},
    "Delhi": {"Business": 1.0, "Sports": 0.9, "Technology": 1.1, "Politics": 1.2, "Entertainment": 1.0},
    "Bangalore": {"Business": 1.1, "Sports": 0.7, "Technology": 1.3, "Politics": 0.9, "Entertainment": 1.0},
    "Pune": {"Business": 1.4, "Sports": 0.9, "Technology": 1.3, "Politics": 0.7, "Entertainment": 0.5}
}

# Streamlit UI
st.title("Newspaper Content Prioritization")

# Sidebar for inputting new articles
st.sidebar.header("Add Article")
title = st.sidebar.text_input("Title")
content = st.sidebar.text_area("Content")
category = st.sidebar.selectbox("Category", options=["Business", "Sports", "Technology", "Politics", "Entertainment"])
likes = st.sidebar.number_input("Likes", min_value=0, value=0)
shares = st.sidebar.number_input("Shares", min_value=0, value=0)
timestamp = st.sidebar.date_input("Timestamp")

if st.sidebar.button("Add Article"):
    if "articles" not in st.session_state:
        st.session_state.articles = []

    st.session_state.articles.append({
        "Title": title,
        "Content": content,
        "Category": category,
        "Likes": likes,
        "Shares": shares,
        "Timestamp": timestamp.strftime('%Y-%m-%d')
    })
    st.sidebar.success("Article added!")

# Select location
st.subheader("Select Location for Publication")
location = st.selectbox("Location", options=LOCATION_WEIGHTS.keys())

# Adjust category weights dynamically
st.subheader("Adjust Global Category Weights")
category_weights = {}
for category, weight in CATEGORY_WEIGHTS.items():
    category_weights[category] = st.slider(f"{category} Weight", 0.0, 1.5, weight, 0.05)

# Process articles if available
if "articles" in st.session_state and st.session_state.articles:
    st.subheader("Input Articles")
    articles_df = pd.DataFrame(st.session_state.articles)

    # Add computed columns
    articles_df['Sentiment'] = articles_df['Content'].apply(get_sentiment_score)
    articles_df['Recency'] = articles_df['Timestamp'].apply(calculate_recency_score)
    articles_df['Engagement'] = (articles_df['Likes'] + articles_df['Shares']) / 1000
    location_weights = LOCATION_WEIGHTS[location]
    articles_df['Priority_Score'] = articles_df.apply(
        calculate_priority,
        axis=1,
        category_weights=category_weights,
        location_weights=location_weights
    )

    # Display processed data
    st.write("### Articles with Computed Scores")
    st.dataframe(articles_df)

    # Sort articles by priority score
    sorted_articles = articles_df.sort_values(by="Priority_Score", ascending=False)

    # Display sorted articles
    st.write(f"### Sorted Articles by Priority for {location}")
    st.dataframe(sorted_articles[['Title', 'Priority_Score', 'Category']])

    # Download sorted articles
    st.download_button(
        label="Download Sorted Articles",
        data=sorted_articles.to_csv(index=False),
        file_name=f"sorted_articles_{location}.csv",
        mime="text/csv"
    )
else:
    st.info("No articles added yet. Use the sidebar to add articles.")
