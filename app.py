"""STEP 3: The website. Run:  python -m streamlit run app.py"""
import streamlit as st
import recommender as rec

st.set_page_config(page_title="Content Personalization", page_icon="🎯")
st.title("🎯 Content Personalization")

users = rec.load_users()
content = rec.load_content()
name = st.sidebar.selectbox("Who are you?", users["name"])
user_id = int(users[users["name"] == name]["user_id"].iloc[0])

# start fresh when the user changes
if st.session_state.get("current_user") != user_id:
    st.session_state["current_user"] = user_id
    for key in ["recs", "viewing", "last_query", "notice"]:
        st.session_state.pop(key, None)


def refresh():
    st.session_state.pop("recs", None)


def act(kind, content_id, shown):
    """Save what the user did, then reload the page."""
    before = set(rec.skipped_categories(user_id))
    rec.record_event(user_id, content_id, kind, shown)
    newly = set(rec.skipped_categories(user_id)) - before
    if newly:
        st.session_state["notice"] = (
            "Hiding " + ", ".join(sorted(newly)) + " for now because you keep picking other topics. "
            "Search it to bring it back.")
    if kind == "view":
        st.session_state["viewing"] = content_id
    refresh()
    st.rerun()


# ---- Sidebar ----
st.sidebar.subheader("Your interests")
st.sidebar.bar_chart(rec.user_interests(user_id))
hidden = rec.skipped_categories(user_id)
if hidden:
    st.sidebar.subheader("Hidden topics")
    for topic in hidden:
        if st.sidebar.button(f"Bring back {topic}", key=f"back_{topic}"):
            rec.reset_topic(user_id, topic)
            refresh()
            st.rerun()

if "notice" in st.session_state:
    st.info(st.session_state.pop("notice"))

# ---- Viewing panel ----
viewing = st.session_state.get("viewing")
if viewing:
    item = content[content["content_id"] == viewing].iloc[0]
    with st.container(border=True):
        st.subheader(item["title"])
        st.caption(f"Topic: {item['category']}")
        st.write(item["description"])
        if st.button("Close", key="close_view"):
            st.session_state.pop("viewing", None)
            st.rerun()

# ---- Search ----
query = st.text_input("🔍 Search content")
if query and query != st.session_state.get("last_query"):
    rec.record_search(user_id, query)
    st.session_state["last_query"] = query
    refresh()
if not query:
    st.session_state["last_query"] = ""

if query:
    results = rec.search_content(query)
    if results.empty:
        st.write("Nothing found.")
    for _, r in results.iterrows():
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{r['title']}** ({r['category']})")
        if c2.button("👁 View", key=f"s{r['content_id']}"):
            act("view", int(r["content_id"]), None)

# ---- Recommendations ----
head1, head2 = st.columns([3, 1])
head1.subheader("Recommended for you")
if head2.button("🔄 New ideas"):
    refresh()
    st.rerun()

if "recs" not in st.session_state:
    st.session_state["recs"] = rec.recommend(user_id, 5, 2)
recs = st.session_state["recs"]
shown = list(recs["category"])

if recs.empty:
    st.write("You have seen everything. Great job!")

for _, r in recs.iterrows():
    cid = int(r["content_id"])
    label = "🎯 Top pick" if r["kind"] == "top" else "🎲 Explore"
    with st.container(border=True):
        st.write(f"**{r['title']}**")
        st.caption(f"{label} · {r['category']} · match {r['score']:.0%} · 👁 {int(r['views'])} views")
        c1, c2, c3 = st.columns(3)
        if c1.button("👁 View", key=f"v{cid}"):
            act("view", cid, shown)
        if c2.button("👍 Like", key=f"l{cid}"):
            act("like", cid, shown)
        if c3.button("👎 Skip", key=f"x{cid}"):
            act("skip", cid, shown)

st.caption("Tip: if you keep picking other topics, the ones you ignore get hidden. Search brings them back.")
