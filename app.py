import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
from itertools import combinations
from collections import Counter

# ------------------------------------------------------
# SAFE NLTK DOWNLOAD
# ------------------------------------------------------
nltk.download("vader_lexicon", quiet=True)

# ------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Ultra-Advanced Text Analysis Suite", layout="wide")
st.title("📊 Advanced Text Analysis Suite for JU Class")

# ------------------------------------------------------
# UPLOAD FILE
# ------------------------------------------------------
uploaded = st.file_uploader("Upload dataset (Excel or CSV)", type=["xlsx", "xls", "csv"])

if uploaded:

    # Load file safely
    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # --------------------------------------------------
    # SIDEBAR SETTINGS
    # --------------------------------------------------
    st.sidebar.header("Settings")

    text_col = st.sidebar.selectbox("Select text column", df.columns)
    model_col = st.sidebar.selectbox("Select model column", df.columns)
    region_col = st.sidebar.selectbox("Select user region column", df.columns)

    # --------------------------------------------------
    # STOPWORDS
    # --------------------------------------------------
    default_stop = {
        "a","about","above","across","after","again","against","all","almost","alone","along","already",
        "also","although","always","am","among","an","and","another","any","anybody","anyone","anything",
        "anyway","anywhere","are","around","as","at","back","be","became","because","become","becomes",
        "been","before","behind","being","below","beside","between","beyond","both","but","by","can",
        "cannot","could","did","do","does","doing","done","down","during","each","either","else",
        "enough","even","ever","every","everybody","everyone","everything","everywhere","except","few",
        "first","five","for","former","formerly","four","from","further","had","has","have","having",
        "he","her","here","hers","herself","him","himself","his","how","however","i","if","in","indeed",
        "instead","into","is","it","its","itself","just","keep","keeps","kept","know","known","last",
        "least","less","let","likely","long","made","make","makes","many","may","maybe","me","might",
        "mine","more","most","mostly","much","must","my","myself","neither","never","new","next","nine",
        "no","nobody","none","nor","not","nothing","now","nowhere","of","off","often","on","once","one",
        "only","onto","or","other","others","otherwise","our","ours","ourselves","out","over","own",
        "part","per","perhaps","put","rather","really","said","same","say","second","see","seem",
        "seemed","seeming","seems","several","she","should","since","six","so","some","somebody",
        "someone","something","sometimes","somewhere","still","such","taking","ten","than","that","the",
        "their","theirs","them","themselves","then","there","therefore","these","they","thing","things",
        "third","this","those","though","three","through","throughout","to","together","too","toward",
        "try","trying","twenty","two","under","until","up","upon","us","use","used","usually","very",
        "via","was","we","well","were","what","whatever","when","whenever","where","whether","which",
        "while","who","whoever","whole","whom","why","will","with","within","without","would","yes",
        "yet","you","your","yours","yourself","yourselves",

        # conversational filler
        "uh","um","hmm","ok","okay","yeah","yep","right","well","basically","literally","actually",
        "kinda","sorta","maybe","guess","just","really","quite",

        # academic filler
        "analysis","study","studies","research","paper","text","section","paragraph","author","authors",

        # punctuation artifacts
        ".", ",", ";", "!", "?", "_", "(", ")", "[", "]", "{", "}", "'", "\""
    }

    sociological_stop = {
        "people","think","say","study","student","analysis","text","use","like",
        "beijing","washington","dc","world","post","post-decarbonized","decarbonized","&"
    }

    custom_stop = st.sidebar.text_area("Add custom stopwords (comma-separated)")
    custom_stop = set(w.strip().lower() for w in custom_stop.split(",") if w.strip())

    stopwords_final = sorted(default_stop.union(sociological_stop).union(custom_stop))

    # --------------------------------------------------
    # Topic modeling sliders
    # --------------------------------------------------
    n_topics = st.sidebar.slider("Number of topic clusters", 3, 20, 6)
    min_df = st.sidebar.slider("Min document frequency", 1, 10, 2)
    max_df = st.sidebar.slider("Max document frequency", 0.1, 1.0, 0.9)
    top_n_words = st.sidebar.slider("Top terms per topic", 5, 30, 10)

    # --------------------------------------------------
    # Create TABS
    # --------------------------------------------------
    tabs = st.tabs([
        "1️⃣ Topic Modeling",
        "2️⃣ STM-style Word Differences",
        "3️⃣ Topic Distance Map",
        "4️⃣ Semantic Network",
        "5️⃣ Sentiment",
        "6️⃣ Wordclouds",
        "7️⃣ Region × Model Analysis"
    ])


    # ------------------------------------------------------
    # TEXT NORMALIZATION FUNCTION (remove punctuation + 1–2 char words)
    # ------------------------------------------------------
    import re
    
    def clean_tokenize(text):
        """
        Full cleaning:
        - lowercase
        - replace non-alphanumeric with spaces
        - split on whitespace
        - remove tokens of length <= 2
        - remove stopwords
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)       # keep only alphanumeric
        tokens = text.split()
        tokens = [t for t in tokens 
                  if len(t) > 2 and t not in stopwords_final]  # remove 1–2 char + stopwords
        return tokens
    
    
    # ------------------------------------------------------
    # CLEAN TEXT
    # ------------------------------------------------------
    df = df.dropna(subset=[text_col])
    
    # cleaned token lists for topic modeling
    df["_clean_tokens"] = df[text_col].astype(str).apply(clean_tokenize)
    
    # join back into text for TF-IDF
    docs = df["_clean_tokens"].apply(lambda toks: " ".join(toks)).tolist()
    
    
    # ------------------------------------------------------
    # TF-IDF (UNIGRAMS ONLY, only clean tokens)
    # ------------------------------------------------------
    tfidf = TfidfVectorizer(
        stop_words=None,          # already cleaned
        max_features=6000,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 1)
    )
    
    X = tfidf.fit_transform(docs)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    
    # ------------------------------------------------------
    # NMF TOPIC MODEL
    # ------------------------------------------------------
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    df["topic"] = W.argmax(axis=1)
    
    
    # ------------------------------------------------------
    # Helper: top words per topic
    # ------------------------------------------------------
    def top_words(topic_idx, n=top_n_words):
        idx = H[topic_idx].argsort()[-n:][::-1]
        return feature_names[idx], H[topic_idx][idx]
    
    
    # ======================================================
    # TAB 1 — TOPIC MODELING
    # ======================================================
    with tabs[0]:
    
        st.subheader("📌 Extracted Topics")
    
        topic_rows = []
        for t in range(n_topics):
            words, _ = top_words(t)
            topic_rows.append({"Topic": t, "Top Terms": ", ".join(words)})
    
        st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)
    
        # ---------------- Topic Distribution
        st.subheader("Topic Distribution by Region")
        fig = px.histogram(
            df, x="topic", color=region_col, barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
        # ---------------- Heatmap
        st.subheader("Topic–Term Heatmap")
        heat = np.vstack([top_words(t, top_n_words)[1] for t in range(n_topics)])
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=heat,
                x=[f"Term {i+1}" for i in range(top_n_words)],
                y=[f"Topic {i}" for i in range(n_topics)],
                colorscale="Viridis"
            )
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    
    
    # ======================================================
    # TAB 2 — STM-STYLE DIFFERENCE PLOTS
    # ======================================================
    with tabs[1]:
    
        st.header("STM-Style Difference-in-Word-Use Analysis")
    
        condition = st.selectbox("Compare by:", [model_col, region_col])
    
        groups = df.groupby(condition)
        if len(groups) != 2:
            st.warning("Select a variable with exactly 2 categories.")
        else:
            g1, g2 = list(groups.groups.keys())
    
            # collect cleaned tokens, NOT raw text
            tokens1 = df[df[condition] == g1]["_clean_tokens"].sum()
            tokens2 = df[df[condition] == g2]["_clean_tokens"].sum()
    
            w1 = Counter(tokens1)
            w2 = Counter(tokens2)
    
            vocab = list(set(w1.keys()).union(set(w2.keys())))
    
            diff = [{"word": w, "diff": w1[w] - w2[w]} for w in vocab]
    
            diff_df = pd.DataFrame(diff)
            diff_df["abs"] = diff_df["diff"].abs()
            diff_df = diff_df.sort_values("abs", ascending=False).head(40)
    
            fig = px.bar(
                diff_df,
                x="word",
                y="diff",
                color="diff",
                color_continuous_scale="RdBu",
                title=f"{g1} vs {g2}: Word Usage Differences"
            )
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # TAB 3 — TOPIC DISTANCE MAP
    # ======================================================
    with tabs[2]:

        st.header("Topic Distance Map (MDS)")
        dist = pairwise_distances(H)
        coords = MDS(
            n_components=2,
            random_state=42,
            dissimilarity="precomputed"
        ).fit_transform(dist)

        fig = px.scatter(
            x=coords[:,0], y=coords[:,1],
            text=[f"T{i}" for i in range(n_topics)],
            color=list(range(n_topics)),
            color_continuous_scale="Viridis"
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # TAB 4 — SEMANTIC NETWORK (GOLD STANDARD + INTERACTIVE)
    # ======================================================
    with tabs[3]:
    
        st.header("Semantic Network (Weighted Strength Centrality + Interactive)")
    
        # -------------------------------------------
        # SLIDERS
        # -------------------------------------------
        min_w = st.slider("Minimum co-occurrence weight", 1, 6, 1)
        min_cent = st.slider("Minimum node strength (centrality filter)", 0.0, 1.0, 0.0, 0.01)
    
        # -------------------------------------------
        # BUILD BASE GRAPH
        # -------------------------------------------
        G = nx.Graph()
    
        for t in range(n_topics):
            words, _ = top_words(t, 12)
            for w1, w2 in combinations(words, 2):
                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += 1
                else:
                    G.add_edge(w1, w2, weight=1)
    
        # -------------------------------------------
        # FILTER EDGES BY WEIGHT
        # -------------------------------------------
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_w]
        G.remove_edges_from(edges_to_remove)
        G.remove_nodes_from(list(nx.isolates(G)))
    
        if len(G.nodes()) == 0:
            st.warning("No nodes remain with this threshold. Lower the filter.")
            st.stop()
    
        # -------------------------------------------
        # WEIGHTED STRENGTH CENTRALITY (GOLD STANDARD)
        # -------------------------------------------
        strength = {
            n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
    
        # Convert to NumPy
        cent_vals = np.array(list(strength.values()), dtype=float)
    
        # Safe normalization
        min_v = np.min(cent_vals)
        max_v = np.max(cent_vals)
        ptp = max_v - min_v
    
        if ptp <= 1e-12:
            cent_norm = np.ones_like(cent_vals) * 0.5
        else:
            cent_norm = (cent_vals - min_v) / ptp
    
        # -------------------------------------------
        # FILTER NODES BY CENTRALITY SLIDER
        # -------------------------------------------
        keep_mask = cent_norm >= min_cent
        keep_nodes = [node for node, keep in zip(G.nodes(), keep_mask) if keep]
    
        G = G.subgraph(keep_nodes).copy()
    
        if len(G.nodes()) == 0:
            st.warning("No nodes remain after centrality filtering.")
            st.stop()
    
        # Recompute strength AFTER filtering
        strength = {
            n: sum(d["weight"] for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
    
        cent_vals = np.array(list(strength.values()), dtype=float)
        min_v = np.min(cent_vals)
        max_v = np.max(cent_vals)
        ptp = max_v - min_v
        if ptp <= 1e-12:
            cent_norm = np.ones_like(cent_vals) * 0.5
        else:
            cent_norm = (cent_vals - min_v) / ptp
    
        # -------------------------------------------
        # COLOR PALETTE — DENSE (shifted by 2)
        # -------------------------------------------
        dense_colors = [
            "rgb(54,14,36)", "rgb(80,20,66)", "rgb(100,31,104)",
            "rgb(113,50,141)", "rgb(119,74,175)", "rgb(120,100,202)",
            "rgb(117,127,221)", "rgb(115,154,228)", "rgb(129,180,227)",
            "rgb(156,201,226)", "rgb(191,221,229)"
        ]
        palette = list(reversed(dense_colors[2:])) # start from 3rd lighter color
    
        # Map centrality → color
        idx = (cent_norm * (len(palette) - 1)).astype(int)
        node_colors = [palette[i] for i in idx]
    
        # -------------------------------------------
        # NODE SIZES
        # -------------------------------------------
        node_sizes = [8 + 80 * c for c in cent_norm]
    
        # -------------------------------------------
        # LAYOUT (anti-overlap tuned spring)
        # -------------------------------------------
        pos = nx.spring_layout(
            G,
            k=0.9,
            iterations=100,
            seed=42,
            weight="weight"
        )
    
        # -------------------------------------------
        # EDGES
        # -------------------------------------------
        edge_x, edge_y = [], []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
    
        # -------------------------------------------
        # NODES
        # -------------------------------------------
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = [f"{n}<br>Strength={strength[n]:.2f}" for n in G.nodes()]
    
        # -------------------------------------------
        # PLOTLY FIGURE
        # -------------------------------------------
        fig_net = go.Figure()
    
        # Edges
        fig_net.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.3, color="dimgray"),
            hoverinfo="none",
            opacity=0.55
        ))
    
        # Nodes
        fig_net.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.97,
                line=dict(color="rgba(0,0,0,0)", width=1),  # nessun bordo fisso
            ),
            
            hoverlabel=dict(
                bgcolor="rgba(255,235,235,0.85)",  # sfondo rosa tenue
                bordercolor="darkred",
                font=dict(color="black")
            ),
        ))
    
        fig_net.update_layout(
            dragmode="pan",
            height=830,
            margin=dict(l=10, r=10, b=10, t=10),
            hovermode="closest"
        )
    
        st.plotly_chart(fig_net, use_container_width=True)
        
    # ======================================================
    # TAB 5 — SENTIMENT (CORRECTED)
    # ======================================================
            
    with tabs[4]:
        
        st.header("Sentiment Analysis (VADER)")
    
        # --- Compute sentiment scores ---
        sia = SentimentIntensityAnalyzer()
    
        df["sentiment_score"] = df[text_col].astype(str).apply(
            lambda x: sia.polarity_scores(x)["compound"]
        )
    
        df["sentiment_label"] = df["sentiment_score"].apply(
            lambda s: 
                "Positive" if s > 0.05 else 
                ("Negative" if s < -0.05 else "Neutral")
        )
    
        # --- Unified color map ---
        color_map = {
            "Positive": "#4DA6FF",   # soft blue
            "Negative": "#FF6666",   # soft red
            "Neutral":  "#BFBFBF"    # soft gray
        }
    
        # =====================================================================
        # 1) Overall Sentiment Distribution (CORRECTED & FIXED)
        # =====================================================================
        st.subheader("Overall Sentiment Distribution")
    
        sent_counts = (
            df["sentiment_label"]
            .value_counts(normalize=True)
            .reset_index()
        )
    
        # Correct column names
        sent_counts.columns = ["sentiment_label", "percent"]
    
        # Ensure numeric
        sent_counts["percent"] = pd.to_numeric(sent_counts["percent"], errors="coerce").fillna(0)
    
        # Display values in %
        sent_counts["percent_display"] = (sent_counts["percent"] * 100).round(1)
    
        # Plot
        fig = px.bar(
            sent_counts,
            x="sentiment_label",
            y="percent",
            color="sentiment_label",
            color_discrete_map=color_map,
            text="percent_display",
        )
    
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            yaxis=dict(ticksuffix="%"),
            xaxis_title="Sentiment category",
            yaxis_title="Percentage of all texts",
        )
    
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Percentages are calculated relative to the entire dataset (not per category).")
    
        # =====================================================================
        # 2) Sentiment by Region  (FIXED — color = sentiment)
        # =====================================================================
        st.subheader("Sentiment by Region")
    
        fig = px.histogram(
            df,
            x=region_col,                  # <-- region on x-axis
            color="sentiment_label",       # <-- colors represent sentiment
            barnorm="percent",             # <-- % within each region
            color_discrete_map=color_map
        )
    
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Percent sentiment within region",
            legend_title="Sentiment"
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        # =====================================================================
        # 3) Sentiment by Model (FIXED — color = sentiment)
        # =====================================================================
        st.subheader("Sentiment by Model")
    
        fig = px.histogram(
            df,
            x=model_col,                   # <-- model on x-axis
            color="sentiment_label",       # <-- colors represent sentiment
            barnorm="percent",             # <-- % within each model
            color_discrete_map=color_map
        )
    
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Percent sentiment within model",
            legend_title="Sentiment"
        )
    
        st.plotly_chart(fig, use_container_width=True)
    # ======================================================
    # TAB 6 — WORDCLOUDS
    # ======================================================
    with tabs[5]:

        st.header("Wordclouds (Region × Model)")

        groups = df.groupby([region_col, model_col])
        for (reg, mod), subset in groups:
            st.subheader(f"{reg} – {mod}")
            text = " ".join(subset[text_col].astype(str))
            if len(text) < 20:
                st.write("Not enough text.")
                continue
            wc = WordCloud(width=1000, height=500,
                           background_color="white").generate(text)
            st.image(wc.to_array(), use_container_width=True)

    # ======================================================
    # TAB 7 — REGION × MODEL INTERACTION
    # ======================================================
    with tabs[6]:

        st.header("Region × Model Interaction")

        fig = px.density_heatmap(
            df, x=region_col, y=model_col,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)




