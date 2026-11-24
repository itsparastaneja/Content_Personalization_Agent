"""The 'brain' used by both the app and the agent.
It can recommend (top picks + random explore picks), search, record views/likes/skips,
and hide topics you keep ignoring.
"""
import random
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
MODEL_PATH = BASE / "model.joblib"
INTERACTIONS_FILE = DATA / "interactions.csv"   # your views / likes / skips
SIGNALS_FILE = DATA / "signals.csv"             # "ignored" and "reset" marks per topic

SKIP_AFTER = 3   # a topic is hidden after you ignore it this many times in a row
COLUMNS = ["content_id", "title", "category", "description", "score", "kind", "views"]


def ensure_setup():
    """If data or model is missing, create them automatically."""
    import subprocess, sys
    content_file = DATA / "content.csv"
    needs_data = not content_file.exists() or not (DATA / "impressions.csv").exists()
    if not needs_data:
        needs_data = "description" not in pd.read_csv(content_file, nrows=0).columns
    if needs_data:
        subprocess.run([sys.executable, str(BASE / "make_data.py")], check=True)
    if needs_data or not MODEL_PATH.exists():
        subprocess.run([sys.executable, str(BASE / "train_model.py")], check=True)


ensure_setup()


def load_content():
    return pd.read_csv(DATA / "content.csv")


def load_users():
    return pd.read_csv(DATA / "users.csv")


def load_history():
    """Old impressions + everything the user did while using the app."""
    history = pd.read_csv(DATA / "impressions.csv")
    if INTERACTIONS_FILE.exists():
        new = pd.read_csv(INTERACTIONS_FILE)[["user_id", "content_id", "category", "clicked"]]
        history = pd.concat([history, new], ignore_index=True)
    return history


def _append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------- topics you keep ignoring ----------
def skipped_categories(user_id):
    """Topics hidden for this user: ignored SKIP_AFTER times since last interest."""
    if not SIGNALS_FILE.exists():
        return []
    signals = pd.read_csv(SIGNALS_FILE)
    signals = signals[signals["user_id"] == user_id]
    hidden = []
    for category, group in signals.groupby("category"):
        kinds = list(group["kind"])
        if "reset" in kinds:                       # interest resets the counter
            last_reset = len(kinds) - 1 - kinds[::-1].index("reset")
            kinds = kinds[last_reset + 1:]
        if kinds.count("ignore") >= SKIP_AFTER:
            hidden.append(category)
    return hidden


def reset_topic(user_id, category):
    """Bring a hidden topic back."""
    _append(SIGNALS_FILE, [{"user_id": user_id, "category": category, "kind": "reset"}])


# ---------- reading ----------
def user_interests(user_id):
    """Click rate per category for one user (higher = likes it more)."""
    history = load_history()
    mine = history[history["user_id"] == user_id]
    table = mine.groupby("category")["clicked"].agg(["sum", "count"])
    table["click_rate"] = (table["sum"] + 1) / (table["count"] + 2)
    return table["click_rate"].sort_values(ascending=False).round(2)


def recommend(user_id, top_n=5, n_explore=2):
    """Return top_n items: mostly best matches + n_explore random 'explore' items."""
    model = joblib.load(MODEL_PATH)
    content = load_content()
    history = load_history()
    mine = history[history["user_id"] == user_id]

    # feature 1: this user's click rate per category
    cat = mine.groupby("category")["clicked"].agg(["sum", "count"])
    cat["rate"] = (cat["sum"] + 1) / (cat["count"] + 2)
    content["user_cat_rate"] = content["category"].map(cat["rate"]).fillna(0.5)

    # feature 2: popularity of each content item (also used as "views")
    pop = history.groupby("content_id")["clicked"].agg(["sum", "count"])
    pop["rate"] = (pop["sum"] + 1) / (pop["count"] + 2)
    content["popularity"] = content["content_id"].map(pop["rate"]).fillna(0.5)
    content["views"] = content["content_id"].map(pop["sum"]).fillna(0).astype(int)

    # don't show things the user already clicked
    already = set(mine[mine["clicked"] == 1]["content_id"])
    content = content[~content["content_id"].isin(already)].copy()

    # hide topics the user keeps ignoring (unless that would leave nothing)
    allowed = content[~content["category"].isin(skipped_categories(user_id))]
    if not allowed.empty:
        content = allowed.copy()

    if content.empty:
        return pd.DataFrame(columns=COLUMNS)

    content["score"] = model.predict_proba(content[["user_cat_rate", "popularity"]])[:, 1]
    content["noisy"] = content["score"] + [random.uniform(0, 0.08) for _ in range(len(content))]

    n_top = max(1, top_n - n_explore)
    top = content.sort_values("noisy", ascending=False).head(n_top).copy()
    top["kind"] = "top"

    rest = content.drop(top.index)
    parts = [top]
    if n_explore > 0 and not rest.empty:
        rest = rest.sample(frac=1)                                  # shuffle
        rest = rest.assign(seen=rest["category"].isin(top["category"]))
        rest = rest.sort_values("seen", kind="stable")              # prefer new topics
        explore = rest.head(n_explore).copy()
        explore["kind"] = "explore"
        parts.append(explore)

    out = pd.concat(parts)
    out["score"] = out["score"].round(2)
    return out[COLUMNS].reset_index(drop=True)


def search_content(query):
    """Find content whose title or category contains the search words."""
    content = load_content()
    q = query.lower().strip()
    mask = (content["title"].str.lower().str.contains(q, regex=False)
            | content["category"].str.lower().str.contains(q, regex=False))
    return content[mask].reset_index(drop=True)


# ---------- writing ----------
def record_event(user_id, content_id, kind="view", shown_categories=None):
    """kind = 'view', 'click' or 'like' (interest)  |  'skip' (not interested).
    shown_categories = topics that were on screen; the ones you did NOT pick count as ignored.
    """
    content = load_content()
    row = content[content["content_id"] == content_id]
    if row.empty:
        raise ValueError(f"No content with id {content_id}")
    category = row.iloc[0]["category"]

    _append(INTERACTIONS_FILE, [{
        "user_id": user_id, "content_id": content_id, "category": category,
        "clicked": 0 if kind == "skip" else 1, "kind": kind,
    }])

    if kind == "skip":
        _append(SIGNALS_FILE, [{"user_id": user_id, "category": category, "kind": "ignore"}])
    else:
        signals = [{"user_id": user_id, "category": category, "kind": "reset"}]
        for other in set(shown_categories or []) - {category}:
            signals.append({"user_id": user_id, "category": other, "kind": "ignore"})
        _append(SIGNALS_FILE, signals)


def record_search(user_id, query):
    """Searching a topic shows interest, so it can't stay hidden."""
    found = search_content(query)
    topics = sorted(set(found["category"]))
    if topics:
        _append(SIGNALS_FILE, [{"user_id": user_id, "category": c, "kind": "reset"} for c in topics])
    return topics
