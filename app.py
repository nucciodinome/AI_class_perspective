import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('vader_lexicon')

st.title("Mini Text Analysis App – Lightweight Version")

uploaded = st.file_uploader("Upload Excel dataset", type=["xlsx"])
if uploaded:
    df = pd.read_excel(uploaded)

    text_col = st.selectbox("Select text column", df.columns)
    model_col = st.selectbox("Select AI model column", df.columns)
    region_col = st.selectbox("Select user region column", df.columns)

    df = df.dropna(subset=[text_col])
    docs = df[text_col].astype(str).tolist()

    if st.button("Run Analysis"):
        st.subheader("Topic Modeling (NMF)")
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(docs)

        nmf = NMF(n_components=5, random_state=42)
        W = nmf.fit_transform(X)
        H = nmf.components_
        df['topic'] = W.argmax(axis=1)

        st.write("Assigned topics:", df['topic'].value_counts())

        st.subheader("Estimate Effect by User Region")
        fig = px.histogram(df, x='topic', color=region_col, barmode='group')
        st.plotly_chart(fig)

        st.subheader("Semantic Network (Co-occurrence of Top Terms)")
        feature_names = tfidf.get_feature_names_out()
        G = nx.Graph()
        for t in range(5):
            top_idx = H[t].argsort()[-5:]
            words = [feature_names[i] for i in top_idx]
            for w1 in words:
                for w2 in words:
                    if w1 != w2:
                        G.add_edge(w1, w2)
        pos = nx.spring_layout(G, seed=42)
        fig_net = plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=800, font_size=8)
        st.pyplot(fig_net)

        st.subheader("Sentiment by AI Model (VADER)")
        sia = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df[text_col].apply(lambda x: sia.polarity_scores(x)['compound'])
        df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
        fig_sent = px.histogram(df, x='sentiment_label', color=model_col)
        st.plotly_chart(fig_sent)

        st.subheader("Word Clouds (Region × Model)")
        groups = df.groupby([region_col, model_col])
        for (reg, mod), subset in groups:
            st.write(f"### {reg} – {mod}")
            text = " ".join(subset[text_col].astype(str).tolist())
            wc = WordCloud(width=600, height=400).generate(text)
            fig_wc = plt.figure(figsize=(6, 4))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig_wc)
