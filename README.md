# Restaurant Reviews Analyzer Using GenAI

**CS 315 – Application Development and Emerging Technologies — Activity 3**

A GenAI-powered Streamlit app that analyzes restaurant reviews: sentiment
analysis and an AI chatbot for asking questions about the dataset — built
following the same 9-step process as the course activity sheet, applied to a
**restaurant reviews** dataset instead of movie reviews.

---

## 1. App Idea

- **Dataset:** restaurant reviews (`data/restaurant_reviews.csv`) — columns:
  `restaurant_name`, `cuisine`, `review_text`, `rating`, `review_date`,
  `food_item` (the specific dish/menu item each review is about — this
  powers the chatbot's "best/worst/famous item" answers).
  Restaurant names are **real, well-known chains and restaurants** — a mix of
  Philippine favorites (Jollibee, Mang Inasal, Max's Restaurant, Manam,
  Locavore, Red Ribbon, Goldilocks, Conti's, Gerry's Grill, Barrio Fiesta,
  Cabalen, Racks, Yellow Cab Pizza, Greenwich, Army Navy, Zark's Burgers,
  Antonio's, and more) and international chains with a strong presence in
  the Philippines (McDonald's, KFC, Shake Shack, Din Tai Fung, Nobu, Olive
  Garden, Starbucks, Krispy Kreme, Tim Ho Wan, Yabu, Ippudo, Marugame Udon,
  Bonchon, Pepper Lunch, Wendy's, Burger King, Popeyes, Texas Chicken,
  Vikings Luxury Buffet, and more). The **review text itself is synthetic
  sample data written for this class exercise** — not scraped from any real
  review site — so the app has realistic-looking data to analyze without
  using anyone's actual review content.
- **Purpose:**
  - Sentiment analysis of each review (Positive / Neutral / Negative) using
    a GenAI API (OpenAI GPT-4), with an offline rule-based fallback.
  - An AI chatbot for asking natural-language questions about the currently
    filtered reviews.

## 2. Project Structure

```
restaurant_review_analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example             # Template for your OpenAI API key
├── README.md                # This file
└── data/
    └── restaurant_reviews.csv   # Sample dataset (86 reviews, 43 restaurants)
```

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate     


pip install -r requirements.txt


cp .env.example .env

export OPENAI_API_KEY=sk-...
```

> If no `OPENAI_API_KEY` is set, the app still runs fully — it automatically
> switches to a lightweight built-in keyword-based sentiment scorer so you
> can demo it without any API cost.

## 4. Load and Clean the Dataset

`app.py` uses Pandas to:
- Strip whitespace from review text and drop empty/missing reviews.
- Parse `review_date` into datetime and drop unparsable rows.
- Coerce `rating` to a clean integer column.

## 5. Integrate GenAI for Analysis

- `genai_sentiment()` sends each review to GPT-4 for a one-word sentiment
  label (Positive/Neutral/Negative).
- A simple GenAI chatbot at the bottom of the app answers natural-language
  questions about the currently filtered dataset. It understands both
  **item-level questions** (e.g. *"what's the best item at Jollibee?"*,
  *"what's KFC's worst dish?"*, *"what is Nobu famous for?"*) using the
  dataset's `food_item` column, and **restaurant-level questions**
  (e.g. *"which restaurant has the most negative reviews?"*,
  *"which restaurant has the highest average rating?"*). It uses GPT-4
  when an API key is set, and a matching rule-based lookup otherwise. The
  offline matcher recognizes common aliases and misspellings (e.g. "mcdonalds"
  without an apostrophe, "in n out" without "Burger") so it isn't limited to
  the exact restaurant name.

## 6. Streamlit Interface

- Sidebar widgets: cuisine filter, restaurant filter, rating range slider.
- Two tabs keep each concern visually separate: **Overview** (metrics +
  reviews table) and **Ask AI** (the chatbot lives in its own tab so it
  doesn't clutter the data view).

## 7. Run the App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## 8. Test and Iterate

Things tested while building this app:
- Filtering to a single restaurant/cuisine to confirm charts update.
- Running with and without `OPENAI_API_KEY` set, to confirm the offline
  fallback works.

Ideas for further iteration:
- Add more reviews / restaurants to the sample dataset.
- Compare GPT-labeled sentiment against the fallback labels for accuracy.

## 9. Deploy Your App

1. Push this folder to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in
   with GitHub.
3. Click **New app**, select the repo/branch, and set `app.py` as the main
   file.
4. Under **Advanced settings → Secrets**, add:
   ```
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Deploy**.

## 10. Next Goals

- [x] Add filters for specific categories (cuisine, restaurant, rating) —
      implemented in the sidebar.
- [x] Include a chatbot for answering user questions about the dataset —
      implemented at the bottom of the app.
- [ ] Add filtering by date range.
- [ ] Support multi-language reviews with automatic translation before
      sentiment analysis.
- [ ] Add a "flag for follow-up" feature for restaurant owners to respond
      to negative reviews directly in the app.
