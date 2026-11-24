# Content Personalization Agent

Recommends content based on what each user clicks and likes, and adapts as they keep using it.

## Easiest way (Windows)
1. Unzip this folder.
2. Double-click `run_all.bat`.
3. Wait. Your browser opens the website.

## Step by step (VS Code terminal)
Open the folder in VS Code, then type these one at a time:

    python -m venv venv
    venv\Scripts\activate
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## The AI chat agent (optional)
1. Get a free key: https://aistudio.google.com/apikey
2. Copy `.env.example`, rename the copy to `.env`, paste your key.
3. Run:  python agent.py

If the model name stops working, change GEMINI_MODEL in `.env`.

## What each file does
- make_data.py   - makes fake users, content and click history
- train_model.py - teaches a model to guess "will this user click this?"
- recommender.py - picks the best items + random explore items, records views/likes/skips, hides ignored topics, searches
- app.py         - the website (Streamlit)
- agent.py       - chat agent that uses the recommender as tools
