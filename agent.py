"""STEP 4 (optional): Chat with an AI agent that uses your recommender.
Needs a free Gemini key in the .env file.  Run:  python agent.py
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import recommender as rec

load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("No GOOGLE_API_KEY found. Copy .env.example to .env and paste your key.")


@tool
def recommend_content(user_id: int, top_n: int = 5) -> str:
    """Get the best content recommendations for a user."""
    cols = ["content_id", "title", "category", "kind", "views"]
    return rec.recommend(user_id, top_n)[cols].to_string(index=False)


@tool
def view_content(user_id: int, content_id: int) -> str:
    """Open a piece of content for a user and show what it says."""
    rec.record_event(user_id, content_id, "view")
    row = rec.load_content().query("content_id == @content_id")
    return row.iloc[0]["description"] if not row.empty else "No such content."


@tool
def like_content(user_id: int, content_id: int) -> str:
    """Record that a user liked a piece of content."""
    rec.record_event(user_id, content_id, "like")
    return "Saved the like."


@tool
def search_content(query: str) -> str:
    """Search content by title or category."""
    found = rec.search_content(query)
    return found.to_string(index=False) if not found.empty else "No results."


@tool
def show_interests(user_id: int) -> str:
    """Show which categories a user likes most."""
    return rec.user_interests(user_id).to_string()


TOOLS = [recommend_content, view_content, like_content, search_content, show_interests]
TOOL_MAP = {t.name: t for t in TOOLS}

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
).bind_tools(TOOLS)


def text_of(message):
    """Gemini sometimes returns a list of parts; join them into one string."""
    c = message.content
    if isinstance(c, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c


def main():
    users = rec.load_users()
    print("Users:", ", ".join(f"{r.user_id}={r.name}" for r in users.itertuples()))
    user_id = int(input("Enter your user id: "))
    messages = [SystemMessage(content=(
        f"You are a friendly content recommendation assistant. The current user has id {user_id}. "
        "Use the tools to recommend, view, search, and record likes. Keep answers short and simple."))]
    print("Type your message (or 'quit').")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        messages.append(HumanMessage(content=text))
        while True:
            ai = llm.invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                print("Agent:", text_of(ai))
                break
            for call in ai.tool_calls:
                result = TOOL_MAP[call["name"]].invoke(call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))


if __name__ == "__main__":
    main()
