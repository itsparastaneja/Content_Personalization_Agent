"""STEP 1: Make fake data (users, content, and who clicked what).
Run:  python make_data.py
"""
import random
from pathlib import Path
import pandas as pd

random.seed(42)
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

# 6 categories x 6 titles = 36 pieces of content
TITLES = {
    "tech": ["How AI Models Learn", "Top 5 Python Tricks", "Phones to Watch in 2027",
             "What is a Neural Network?", "Cloud Computing in Plain Words", "Build Your First Website"],
    "sports": ["Cricket Finals Recap", "Football Transfer News", "Beginner Running Plan",
               "Tennis Grand Slam Highlights", "Best Gym Routines", "Chess Openings Guide"],
    "music": ["Top Indie Songs This Week", "Learn Guitar in 30 Days", "History of Rock",
              "Lo-fi Beats for Studying", "Best Concerts Coming Up", "How Songs Get Made"],
    "food": ["Easy 15-Minute Dinners", "Street Food of Delhi", "Perfect Masala Chai",
             "Healthy Breakfast Ideas", "Baking Bread at Home", "Best Biryani Spots"],
    "travel": ["Weekend Trips from Gurgaon", "Budget Trip to Goa", "Hidden Gems in Rajasthan",
               "Packing Like a Pro", "Mountain Treks for Beginners", "Cheap Flights Guide"],
    "finance": ["Saving Money as a Student", "What is a Mutual Fund?", "Budgeting in 3 Steps",
                "Understanding Credit Scores", "Side Income Ideas", "Stock Market Basics"],
}

# --- content.csv ---
rows = []
content_id = 1
for category, titles in TITLES.items():
    for title in titles:
        description = (f"A quick, friendly read: '{title}'. It covers the basics of {category} "
                       f"in simple words, with easy examples you can try today.")
        rows.append({"content_id": content_id, "title": title,
                     "category": category, "description": description})
        content_id += 1
content = pd.DataFrame(rows)
content.to_csv(DATA / "content.csv", index=False)

# --- users.csv ---
names = ["Aarav", "Diya", "Rohan", "Meera", "Kabir", "Isha", "Vihaan", "Anaya", "Arjun", "Sara"]
users = pd.DataFrame({"user_id": range(1, len(names) + 1), "name": names})
users.to_csv(DATA / "users.csv", index=False)

# --- impressions.csv --- (each row: a user was SHOWN an item, did they click?)
# Secret rules used to FAKE the data (the model must discover these on its own):
#   - every user secretly loves 2 categories
#   - some items are just generally more popular
categories = list(TITLES.keys())
secret_favorites = {uid: random.sample(categories, 2) for uid in users["user_id"]}
item_quality = {cid: random.uniform(0.0, 0.2) for cid in content["content_id"]}

impressions = []
for uid in users["user_id"]:
    shown = content.sample(n=60, replace=True, random_state=uid)
    for _, item in shown.iterrows():
        p = 0.65 if item["category"] in secret_favorites[uid] else 0.08
        p += item_quality[item["content_id"]]
        clicked = 1 if random.random() < p else 0
        impressions.append({"user_id": uid, "content_id": item["content_id"],
                            "category": item["category"], "clicked": clicked})

pd.DataFrame(impressions).to_csv(DATA / "impressions.csv", index=False)

# start with no new events
for name in ["events.csv", "interactions.csv", "signals.csv"]:
    old = DATA / name
    if old.exists():
        old.unlink()

print(f"Done! Made {len(content)} content items, {len(users)} users, {len(impressions)} impressions.")
print("Files are in the 'data' folder.")
