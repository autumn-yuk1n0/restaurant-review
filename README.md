# Restaurant Reviews Analyzer Using GenAI

**CS 315 – Application Development and Emerging Technologies — Activity 3**

A GenAI-powered Streamlit app that analyzes restaurant reviews: sentiment
analysis and an AI chatbot for asking questions about the dataset — built
following the same 9-step process as the course activity sheet, applied to a
**restaurant reviews** dataset instead of movie reviews.

---

## 1. Purpose

The app takes a dataset of restaurant reviews and turns it into something
explorable: filter it by cuisine, restaurant, or rating, see it summarized
with GenAI-generated sentiment labels, and ask it natural-language questions
("what's the best item at Jollibee?") through a built-in chatbot. It's meant
to demonstrate a full GenAI integration — a real model call, an automatic
offline fallback when no API key is configured, and an interactive UI — on a
realistic, food-focused dataset.

## 2. Dataset

- **File:** `data/restaurant_reviews.csv`
- **Columns:** `restaurant_name`, `cuisine`, `review_text`, `rating`,
  `review_date`, `food_item` (the specific dish/menu item each review is
  about — this powers the chatbot's "best/worst/famous item" answers).
- Restaurant names are **real, well-known chains and restaurants** — a mix of
  Philippine favorites (Jollibee, Mang Inasal, Max's Restaurant, Manam,
  Locavore, Red Ribbon, Goldilocks, Conti's, Gerry's Grill, Barrio Fiesta,
  Cabalen, Racks, Yellow Cab Pizza, Greenwich, Army Navy, Zark's Burgers,
  Antonio's, and more) and international chains with a strong presence in
  the Philippines (McDonald's, KFC, Shake Shack, Din Tai Fung, Nobu, Olive
  Garden, Starbucks, Krispy Kreme, Tim Ho Wan, Yabu, Ippudo, Marugame Udon,
  Bonchon, Pepper Lunch, Wendy's, Burger King, Popeyes, Texas Chicken,
  Vikings Luxury Buffet, and more).
- The **review text itself is synthetic sample data** written for this class
  exercise — not scraped from any real review site — so the app has
  realistic-looking data to analyze without using anyone's actual review
  content. (86 reviews across 43 restaurants.)

## 3. Sentiment Analysis

Every (filtered) review is labeled **Positive**, **Neutral**, or **Negative**
by `genai_sentiment()`, which picks a backend automatically:

1. **OpenAI (GPT-4)** — used if `OPENAI_API_KEY` is set. Sends the review
   text and asks for a one-word sentiment label.
2. **Offline rule-based fallback** — used if no key is set, or if a live
   call fails. A lightweight keyword scorer (`fallback_sentiment()`) counts
   positive- and negative-leaning words in the review text so the app still
   runs end-to-end with zero API cost or setup.

This means the app never breaks for lack of a key — it just degrades
gracefully to the offline method.

## 4. GenAI Chatbot ("Ask AI")

A chatbot at the bottom of the app answers natural-language questions about
the **currently filtered** dataset. It understands:

- **Item-level questions** — e.g. *"what's the best item at Jollibee?"*,
  *"what's KFC's worst dish?"*, *"what is Nobu famous for?"* — using the
  `food_item` column.
- **Restaurant-level questions** — e.g. *"which restaurant has the most
  negative reviews?"*, *"which restaurant has the highest average rating?"*

It uses GPT-4 when `OPENAI_API_KEY` is set, and a rule-based lookup
otherwise. The offline matcher recognizes common aliases and misspellings
(e.g. "mcdonalds" without an apostrophe, "in n out" without "Burger") so
it isn't limited to the exact restaurant name.

## 5. Project Structure

```
restaurant_review_analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example             # Template for your API key(s)
├── README.md                # This file
└── data/
    └── restaurant_reviews.csv   # Sample dataset (86 reviews, 43 restaurants)
```

## 6. Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Enable real GenAI sentiment analysis
export OPENAI_API_KEY=sk-...
```

> Never commit a real API key to the repo or paste it into a chat/document —
> always set it as an environment variable (or a Streamlit Cloud secret, see
> §10) so it stays out of source control.

> If no key is set, the app still runs fully — it automatically switches to
> the offline keyword-based sentiment scorer so you can demo it without any
> API cost.

## 7. Load and Clean the Dataset

`app.py` uses Pandas to:
- Strip whitespace from review text and drop empty/missing reviews.
- Parse `review_date` into datetime and drop unparsable rows.
- Coerce `rating` to a clean integer column.

## 8. Streamlit Interface

- **Sidebar widgets:** cuisine filter, restaurant filter, rating range
  slider.
- **Two tabs:** **Overview** (metrics — total reviews, average rating, %
  positive sentiment — plus the full reviews table with sentiment labels)
  and **Ask AI** (the chatbot, kept in its own tab so it doesn't clutter the
  data view).

## 9. Test and Iterate

Things tested while building this app:
- Filtering to a single restaurant/cuisine to confirm the table and metrics
  update.
- Running with and without `OPENAI_API_KEY` set, to confirm the offline
  fallback works.

Ideas for further iteration:
- Add more reviews / restaurants to the sample dataset.
- Compare GPT-labeled sentiment against the offline fallback labels for
  accuracy.
- Bring back a visual sentiment/trend/keyword breakdown as a separate,
  optional tab.

## 10. Deploy Your App

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

## 11. Next Goals

- [x] Add filters for specific categories (cuisine, restaurant, rating) —
      implemented in the sidebar.
- [x] Include a chatbot for answering user questions about the dataset —
      implemented in the Ask AI tab.
- [ ] Add filtering by date range.
- [ ] Support multi-language reviews with automatic translation before
      sentiment analysis.
- [ ] Add a "flag for follow-up" feature for restaurant owners to respond
      to negative reviews directly in the app.
