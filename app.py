import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

try:
    from sentence_transformers import SentenceTransformer
    has_sbert = True
except ImportError:
    has_sbert = False

st.set_page_config(page_title="Social Signal Clustering", layout="wide")

st.title("Interactive Social Signal Clustering Dashboard")
st.write("""
This dashboard groups similar social media signals using KMeans clustering and keywords OR advanced sentence embeddings (if available).
Upload a CSV, adjust cluster count, and explore trending topics!
""")

# Step 1: Data Input
uploaded = st.file_uploader("Upload CSV (column: 'text' or select below):", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("First 5 rows of your data:")
    st.write(df.head())
    if 'text' not in df.columns:
        text_col = st.selectbox("Select column with social signal text:", df.columns)
    else:
        text_col = 'text'
    texts = df[text_col].fillna('').astype(str).values.tolist()
else:
    # Example data
    texts = [
        "Kaplan delivers unity speech under floral arch",
        "New mandate period for city council announced",
        "Election update: results bring new leadership",
        "Church board members chosen after prayer vote",
        "Flower arch at city hall symbolizes reform",
        "Faith and democracy discussed at city meeting",
    ]
    st.info("No data uploaded. Using example social signals.")

st.write(f"Total posts: **{len(texts)}**")
num_clusters = st.slider("Number of clusters/topics:", 2, min(10, len(texts)), 3)

# Step 2: Choose clustering method
method = st.radio("Clustering method:", [
    "TF-IDF (classic, faster)",
    "Sentence Embeddings (smarter, needs package)",
])
if method == "Sentence Embeddings (smarter, needs package)" and not has_sbert:
    st.warning("SentenceTransformer not installed, falling back to TF-IDF.")
    method = "TF-IDF (classic, faster)"

# Step 3: Vectorize and Cluster
if method == "TF-IDF (classic, faster)":
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X = model.encode(texts)

km = KMeans(n_clusters=num_clusters, random_state=42)
labels = km.fit_predict(X)

# Step 4: Display clusters
tab1, tab2 = st.tabs(["Clustered Signals", "Trending Topics per Cluster"])

with tab1:
    st.subheader("Signals Grouped by Topic")
    for i in range(num_clusters):
        st.markdown(f"### Cluster {i+1}")
        cluster_posts = [t for t, l in zip(texts, labels) if l == i]
        for post in cluster_posts:
            st.write("- " + post)
        st.write(f"*Total in cluster: {len(cluster_posts)}*")

with tab2:
    st.subheader("Top Keywords per Cluster (Trending Topics)")
    for i in range(num_clusters):
        cluster_posts = [t for t, l in zip(texts, labels) if l == i]
        # Get keywords with TF vectorizer even if using embeddings for clustering
        vec = TfidfVectorizer(stop_words='english', max_features=20)
        if len(cluster_posts) > 1:
            Xc = vec.fit_transform(cluster_posts)
            keywords = vec.get_feature_names_out()
            scores = Xc.sum(axis=0).A1
            sorted_idx = scores.argsort()[::-1]
            st.markdown(f"#### Cluster {i+1} Top Topics")
            st.write(", ".join([keywords[j] for j in sorted_idx[:6]]))
            # Optional: Plot bar
            fig, ax = plt.subplots()
            ax.bar([keywords[j] for j in sorted_idx[:6]],[scores[j] for j in sorted_idx[:6]])
            ax.tick_params(axis='x', rotation=30)
            st.pyplot(fig)
        else:
            st.write("Not enough posts for keyword extraction.")

# (Optional) GPT/LLM for summarizing clusters
use_gpt = st.checkbox("Summarize clusters with GPT (requires OpenAI key)", value=False)
if use_gpt:
    import openai
    openai.api_key = st.text_input("Paste your OpenAI API key:", type="password")
    if openai.api_key:
        st.subheader("Cluster Summaries by GPT")
        for i in range(num_clusters):
            cluster_posts = [t for t, l in zip(texts, labels) if l == i]
            prompt = "Summarize in one phrase the main topic in these posts: " + "\n".join(cluster_posts)
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0)
                summary = completion.choices[0].message['content']
                st.markdown(f"**Cluster {i+1}:** {summary}")
            except Exception as e:
                st.error(f"Error calling GPT: {e}")

# Footer
st.markdown("""
---
<small>
Created for multi-platform social trend detection.
Powered by Streamlit + scikit-learn + optional OpenAI/GPT.<br>
<a href="https://streamlit.io" target="_blank">Build your own with Streamlit!</a>
</small>
""", unsafe_allow_html=True)
