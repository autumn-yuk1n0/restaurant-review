import os
import re

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Restaurant Reviews Analyzer",
    page_icon="🍽️",
    layout="wide",
)

DATA_PATH = os.path.join("data", "restaurant_reviews.csv")


USE_OPENAI = False
try:
    from openai import OpenAI

    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI()
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False


POSITIVE_WORDS = {
    "amazing", "great", "delicious", "perfect", "perfectly", "loved", "love",
    "fresh", "friendly", "generous", "juicy", "elegant", "flavorful", "best",
    "favorite", "attentive", "smoky", "buttery", "crispy", "creamy", "cozy",
}
NEGATIVE_WORDS = {
    "slow", "overpriced", "cold", "bland", "expensive", "tiny", "dry",
    "oily", "off", "smelly", "watery", "forgot", "long", "wait", "fishy",
}


def fallback_sentiment(text: str) -> str:
    """Very small keyword-based sentiment fallback (used if no GenAI key)."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    pos = sum(w in POSITIVE_WORDS for w in words)
    neg = sum(w in NEGATIVE_WORDS for w in words)
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def genai_sentiment(text: str) -> str:
    """Calls GPT for sentiment classification; falls back if unavailable."""
    if not USE_OPENAI:
        return fallback_sentiment(text)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the sentiment of the restaurant review as "
                        "exactly one word: Positive, Negative, or Neutral."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=3,
            temperature=0,
        )
        label = response.choices[0].message.content.strip()
        if label not in {"Positive", "Negative", "Neutral"}:
            return fallback_sentiment(text)
        return label
    except Exception:
        return fallback_sentiment(text)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleaning logic used for both the sample dataset and uploads."""
    df = df.copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df.dropna(subset=["review_text", "rating"])
    df = df[df["review_text"].str.len() > 0]
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "review_date"])
    df["rating"] = df["rating"].astype(int)
    if "cuisine" not in df.columns:
        df["cuisine"] = "Unknown"
    if "food_item" not in df.columns:
        df["food_item"] = "the dish"
    return df.reset_index(drop=True)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return clean_data(pd.read_csv(path))


@st.cache_data
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment"] = df["review_text"].apply(genai_sentiment)
    return df



st.title("🍽️ Restaurant Reviews Analyzer Using GenAI")
st.caption(
    "Explore the sample dataset of restaurant reviews with GenAI-powered "
    "sentiment analysis, common keyword extraction, and trend visualization."
)

if not USE_OPENAI:
    st.info(
        "No `OPENAI_API_KEY` detected — running with the built-in offline "
        "sentiment fallback so the app still works. Set the environment "
        "variable to use real GPT-4 sentiment analysis.",
        icon="ℹ️",
    )

df = load_data(DATA_PATH)

with st.sidebar:
    st.header("🔎 Filters")
    cuisines = sorted(df["cuisine"].dropna().unique())
    selected_cuisines = st.multiselect("Cuisine", cuisines, default=cuisines)

    restaurants = sorted(df["restaurant_name"].dropna().unique())
    selected_restaurants = st.multiselect(
        "Restaurant", restaurants, default=restaurants
    )

    min_rating, max_rating = st.slider("Rating range", 1, 5, (1, 5))

filtered_df = df[
    df["cuisine"].isin(selected_cuisines)
    & df["restaurant_name"].isin(selected_restaurants)
    & df["rating"].between(min_rating, max_rating)
]

st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** reviews")


if filtered_df.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

with st.spinner("Running GenAI sentiment analysis on reviews..."):
    result_df = analyze_sentiment(filtered_df)


tab1, tab5 = st.tabs(
    ["📊 Overview", "💬 Ask AI"]
)

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(result_df))
    col2.metric("Average Rating", f"{result_df['rating'].mean():.2f} ⭐")
    pos_pct = (result_df["sentiment"] == "Positive").mean() * 100
    col3.metric("% Positive Sentiment", f"{pos_pct:.0f}%")

    st.subheader("Reviews Table")
    display_cols = [
        c for c in ["restaurant_name", "cuisine", "food_item", "rating", "sentiment", "review_text", "review_date"]
        if c in result_df.columns
    ]
    st.dataframe(
        result_df[display_cols].sort_values("review_date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def answer_question(question: str, data: pd.DataFrame) -> str:
    """Answers questions about the dataset. Uses GPT if available; otherwise
    a rule-based matcher that understands per-restaurant "best/worst/famous
    item" questions as well as overall sentiment/rating questions."""
    if USE_OPENAI:
        try:
            context = data[
                ["restaurant_name", "food_item", "rating", "sentiment", "review_text"]
            ].to_csv(index=False)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer questions about the following restaurant "
                            "review dataset (CSV, includes a food_item column "
                            "naming the dish each review is about). Be concise.\n\n"
                            + context
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass


    if data.empty:
        return "There's no data in the current filters to answer that."

    def normalize(s: str) -> str:
        s = s.lower().replace("'", "")
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return " ".join(s.split())

    q = normalize(question)

    ALIASES = {
        "Jollibee": ["jollibee"],
        "Mang Inasal": ["mang inasal"],
        "Manam Comfort Filipino": ["manam"],
        "Max's Restaurant": ["maxs restaurant", "maxs", "max"],
        "Chowking": ["chowking"],
        "Locavore": ["locavore"],
        "Sentro 1771": ["sentro 1771", "sentro"],
        "Pancake House": ["pancake house"],
        "McDonald's": ["mcdonalds", "mcdo", "mcd"],
        "KFC": ["kfc"],
        "Shake Shack": ["shake shack"],
        "Din Tai Fung": ["din tai fung", "dtf"],
        "Nobu": ["nobu"],
        "The Cheesecake Factory": ["cheesecake factory", "cheesecake"],
        "Pizza Hut": ["pizza hut"],
        "In-N-Out Burger": ["in n out", "innout"],
        "Olive Garden": ["olive garden"],
        "Antonio's": ["antonios", "antonio"],
        "Red Ribbon": ["red ribbon"],
        "Goldilocks": ["goldilocks"],
        "Conti's": ["contis", "conti"],
        "Gerry's Grill": ["gerrys grill", "gerrys"],
        "Barrio Fiesta": ["barrio fiesta"],
        "Cabalen": ["cabalen"],
        "Racks": ["racks"],
        "Yellow Cab Pizza": ["yellow cab"],
        "Greenwich": ["greenwich"],
        "Army Navy": ["army navy"],
        "Zark's Burgers": ["zarks burgers", "zarks"],
        "Kenny Rogers Roasters": ["kenny rogers"],
        "Starbucks": ["starbucks"],
        "Krispy Kreme": ["krispy kreme"],
        "Tim Ho Wan": ["tim ho wan"],
        "Yabu": ["yabu"],
        "Ippudo": ["ippudo"],
        "Marugame Udon": ["marugame udon", "marugame"],
        "Bonchon": ["bonchon"],
        "Pepper Lunch": ["pepper lunch"],
        "Wendy's": ["wendys"],
        "Burger King": ["burger king"],
        "Popeyes": ["popeyes"],
        "Texas Chicken": ["texas chicken"],
        "Vikings Luxury Buffet": ["vikings"],
    }

    matched_restaurant = None
    for name in data["restaurant_name"].unique():
        aliases = ALIASES.get(name, [normalize(name)])
        if any(f" {alias} " in f" {q} " for alias in aliases):
            matched_restaurant = name
            break

    if matched_restaurant:
        subset = data[data["restaurant_name"] == matched_restaurant]
        if subset.empty:
            return f"No reviews for '{matched_restaurant}' in the current filters."

        if any(k in q for k in ["worst", "bad ", "least liked"]):
            row = subset.loc[subset["rating"].idxmin()]
            return (
                f"At {matched_restaurant}, the lowest-rated item mentioned is "
                f"'{row['food_item']}' ({row['rating']}⭐): \"{row['review_text']}\""
            )

        if any(k in q for k in ["famous", "popular", "known for", "signature", "specialty"]):
            counts = subset["food_item"].value_counts()
            top_item = counts.idxmax()
            if counts.max() == 1:  
                top_item = subset.loc[subset["rating"].idxmax(), "food_item"]
            avg_r = subset.loc[subset["food_item"] == top_item, "rating"].mean()
            return (
                f"The most talked-about item at {matched_restaurant} is "
                f"'{top_item}' (avg rating {avg_r:.1f}⭐)."
            )

        if any(k in q for k in ["best", "good ", "recommend", "should i get", "should i order"]):
            row = subset.loc[subset["rating"].idxmax()]
            return (
                f"At {matched_restaurant}, the highest-rated item mentioned is "
                f"'{row['food_item']}' ({row['rating']}⭐): \"{row['review_text']}\""
            )

        
        avg = subset["rating"].mean()
        return (
            f"{matched_restaurant} has an average rating of {avg:.1f}⭐ across "
            f"{len(subset)} review(s) in the current filter."
        )

   
    if any(k in q for k in ["best food", "best item", "best dish", "top rated dish"]):
        row = data.loc[data["rating"].idxmax()]
        return (
            f"Across all restaurants, the top-rated item is '{row['food_item']}' "
            f"at {row['restaurant_name']} ({row['rating']}⭐)."
        )
    if any(k in q for k in ["worst food", "worst item", "worst dish"]):
        row = data.loc[data["rating"].idxmin()]
        return (
            f"Across all restaurants, the lowest-rated item is '{row['food_item']}' "
            f"at {row['restaurant_name']} ({row['rating']}⭐)."
        )
    if "negative" in q and "restaurant" in q:
        counts = data[data["sentiment"] == "Negative"]["restaurant_name"].value_counts()
        if counts.empty:
            return "There are no negative reviews in the current filtered data."
        return f"'{counts.idxmax()}' has the most negative reviews in the current filtered data."
    if "positive" in q and "restaurant" in q:
        counts = data[data["sentiment"] == "Positive"]["restaurant_name"].value_counts()
        if counts.empty:
            return "There are no positive reviews in the current filtered data."
        return f"'{counts.idxmax()}' has the most positive reviews in the current filtered data."
    if "average rating" in q or "highest rated" in q:
        best_avg = data.groupby("restaurant_name")["rating"].mean().idxmax()
        return f"'{best_avg}' has the highest average rating."

    return (
        "Try asking things like: \"what's the best item at Jollibee?\", "
        "\"what's the worst dish at KFC?\", \"what's Nobu famous for?\", or "
        "\"which restaurant has the highest average rating?\" "
        "(Connect an OPENAI_API_KEY for fully open-ended answers.)"
    )


with tab5:
    st.subheader("💬 Ask a question about these reviews")
    st.caption(
        "e.g. \"What's the best item at Jollibee?\", \"What's KFC's worst dish?\", "
        "\"What is Nobu famous for?\", or \"Which restaurant has the highest average rating?\""
    )

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_question = st.chat_input("Ask about a restaurant or dish...")
    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        answer = answer_question(user_question, result_df)
        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            st.write(answer)

st.divider()
st.caption(
    "Activity 3 — Restaurant Reviews Analyzer Using GenAI | "
    "Built with Streamlit + Pandas + OpenAI GPT-4 (with offline fallback)"
)
