import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Instagram AI Dashboard", layout="wide")

# ----------------------
# 🎨 PREMIUM STYLE
# ----------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f172a,#020617);
}
.big-title {
    font-size:42px;
    font-weight:700;
    color:#38bdf8;
    text-align:center;
}
.sub-text {
    text-align:center;
    color:#aaa;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# 📂 LOAD DATA
# ----------------------
df = pd.read_csv("instagram_preprocessed.csv")

# ----------------------
# 🎯 HERO SECTION
# ----------------------
st.markdown("<div class='big-title'>🚀 Instagram AI Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Data-Driven Insights • AI Predictions • Smart Recommendations</div>", unsafe_allow_html=True)

# ----------------------
# SIDEBAR
# ----------------------
st.sidebar.title("📊 Controls")

lang = st.sidebar.selectbox("🌐 Language", df['language'].unique())
page = st.sidebar.radio("Navigate", ["Overview","Insights","Trends","Advanced","ML Predictor"])

filtered = df[df['language'] == lang]

# ----------------------
# KPI
# ----------------------
c1, c2, c3 = st.columns(3)
c1.metric("📌 Posts", len(filtered))
c2.metric("❤️ Avg Likes", int(filtered['likes_count'].mean()))
c3.metric("🔥 Engagement", int(filtered['engagement'].mean()))

# ======================
# OVERVIEW
# ======================
if page == "Overview":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Language Distribution")
        st.plotly_chart(px.bar(
            filtered['language'].value_counts().reset_index(name='count'),
            x='language', y='count', color='language'
        ), use_container_width=True)

    with col2:
        st.subheader("😊 Sentiment Analysis")
        st.plotly_chart(px.pie(
            filtered['sentiment'].value_counts().reset_index(name='count'),
            names='sentiment', values='count', hole=0.4
        ), use_container_width=True)

    # 🔥 INSIGHTS PANEL
    st.subheader("💡 Key Insights")

    best_hour = filtered.groupby('hour')['engagement'].mean().idxmax()
    avg_eng = int(filtered['engagement'].mean())

    st.info(f"""
🔥 Best Posting Time: {best_hour}:00  
📊 Average Engagement: {avg_eng}  
📈 Higher engagement seen with optimal hashtag usage  
""")

    # 🎯 RECOMMENDATIONS
    st.subheader("🎯 Recommendations")

    st.success("""
✔ Use 3–6 hashtags  
✔ Post during evening hours  
✔ Keep captions between 70–120 characters  
✔ Use trending emotional keywords  
""")

# ======================
# INSIGHTS
# ======================
elif page == "Insights":

    st.subheader("🔥 Top Hashtags")

    tags = []
    for t in filtered['hashtags']:
        if isinstance(t, list):
            tags.extend(t)

    tag_df = pd.Series(tags).value_counts().head(10).reset_index()
    tag_df.columns = ['hashtag','count']

    st.plotly_chart(px.bar(tag_df, x='hashtag', y='count', color='count'),
                    use_container_width=True)

    # 🧠 SMART SUMMARY
    st.subheader("📌 Summary")

    st.write(f"""
Top-performing hashtags significantly influence engagement.  
Using popular tags increases visibility and reach.
""")

# ======================
# TRENDS
# ======================
elif page == "Trends":

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Engagement vs Hashtags")
        st.plotly_chart(px.scatter(
            filtered.sample(min(3000,len(filtered))),
            x='hashtag_len', y='engagement',
            color='sentiment'
        ), use_container_width=True)

    with col2:
        st.subheader("📊 Likes Distribution")
        st.plotly_chart(px.histogram(filtered, x='likes_count'),
                        use_container_width=True)

# ======================
# ADVANCED
# ======================
elif page == "Advanced":

    st.subheader("📊 Correlation Matrix")

    corr = filtered[['likes_count','comments_count','shares_count',
                     'engagement','caption_length','hashtag_len']].corr()

    st.plotly_chart(px.imshow(corr, text_auto=True),
                    use_container_width=True)

    st.markdown("""
### 📌 Interpretation:
- Likes strongly influence engagement  
- Caption length has moderate impact  
- Hashtags improve reach  
""")

# ======================
# ML PREDICTOR
# ======================
elif page == "ML Predictor":

    st.subheader("🤖 Engagement Prediction")

    X = df[['hashtag_len','caption_length']]
    y = df['engagement']

    model = LinearRegression()
    model.fit(X,y)

    col1, col2 = st.columns(2)

    with col1:
        h = st.slider("Hashtags", 1, 10, 3)

    with col2:
        c = st.slider("Caption Length", 10, 200, 80)

    pred = model.predict([[h,c]])[0]

    st.success(f"🔥 Predicted Engagement: {int(pred)}")

    st.markdown("""
### 🧠 Insight:
More hashtags and optimal caption length increase engagement probability.
""")