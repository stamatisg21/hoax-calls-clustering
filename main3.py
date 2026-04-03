# hoax_analysis_pipeline_v2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

try:
    from umap import UMAP
except ImportError:
    raise ImportError("Please install umap-learn: pip install umap-learn")

DetectorFactory.seed = 0


import re
#import spacy
from nltk.corpus import stopwords

# -----------------------------
# improved STOPWORDS / preprocess_text
# -----------------------------
# Keep single-word stopwords in a set for O(1) checks.
SINGLE_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    # ... (rest of single words from your set)
    "ems"
}
# Keep multiword phrases separate (lowercased)
PHRASE_STOPWORDS = [
    "call now", "free consultation", "limited time",
    "important message", "urgent", "hello just checking",
    "just checking account", "account hello just",
    "press option", "checking account", "hello"
]

def preprocess_text(text):
    text = str(text).lower()

    # 1) Normalize phone/urls/numbers
    text = re.sub(r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b", " <PHONE> ", text)
    text = re.sub(r"\b\d{5,}\b", " <NUM> ", text)
    text = re.sub(r"\$\d+(\.\d+)?", " <MONEY> ", text)
    text = re.sub(r"http\S+|www\S+", " <URL> ", text)

    # 2) Remove multi-word stop phrases first (so phrases like "call now" are removed)
    for phrase in PHRASE_STOPWORDS:
        # escape phrase to safely insert into regex
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)

    # 3) Remove punctuation (but keep <> tokens) and normalize whitespace
    text = re.sub(r"[^\w\s<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 4) Tokenize & remove single-word stopwords
    tokens = [t for t in text.split() if t not in SINGLE_STOPWORDS]

    return " ".join(tokens)



# -----------------------------
# Language Detection
# -----------------------------
def detect_language(text):
    try:
        return detect(str(text))
    except:
        return "unknown"

# -----------------------------
# Load, filter, and prepare data
# -----------------------------

def load_and_prepare_data(path_911, path_meta, drop_non_en=True, simulate_benign=True):
    df_911 = pd.read_csv(path_911)
    df_meta = pd.read_csv(path_meta)
    df_911 = df_911.reset_index(drop=True)
    df_meta = df_meta.reset_index(drop=True)

    df = pd.concat([df_911.head(len(df_meta)), df_meta], axis=1)

    # Detect language
    df["language_detected"] = df["transcript"].apply(detect_language)
    if drop_non_en:
        df = df[df["language_detected"] == "en"].reset_index(drop=True)
    print(f"[INFO] After filtering, {len(df)} English rows remain")

    # Combine text columns
    df["full_text"] = df["desc"].astype(str) + " " + df["title"].astype(str) + " " + df["transcript"].astype(str)

    # Optional: simulate small benign dataset
    if simulate_benign:
        n_synthetic = min(50, len(df)//5)
        benign_texts = ["Hello, just checking in about your account."] * n_synthetic
        df_benign = pd.DataFrame({
            "desc": ["Synthetic"]*n_synthetic,
            "title": ["Benign"]*n_synthetic,
            "transcript": benign_texts,
            "language_detected": ["en"]*n_synthetic,
            "full_text": benign_texts
        })
        df = pd.concat([df, df_benign], ignore_index=True)
        print(f"[INFO] Added {n_synthetic} synthetic benign rows")
    return df

# -----------------------------
# Compute SBERT Embeddings
# -----------------------------
def compute_embeddings(df, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["full_text"].tolist(), batch_size=32, show_progress_bar=True)
    return embeddings

from sklearn.metrics import pairwise_distances

# -----------------------------
# Improved Clustering with HDBSCAN (cosine fix)
# -----------------------------
from sklearn.metrics import pairwise_distances

def cluster_embeddings(embeddings, min_cluster_size=25, min_samples=3, n_components=5, use_cosine=True):
    """
    Run HDBSCAN on SBERT embeddings with optional UMAP reduction.
    Supports cosine distance by precomputing the distance matrix (float64).
    """
    print("[INFO] Reducing embeddings with UMAP...")
    reducer = UMAP(n_components=n_components, random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings)

    if use_cosine:
        print("[INFO] Computing cosine distance matrix...")
        distance_matrix = pairwise_distances(embeddings_reduced, metric="cosine").astype(np.float64)
        metric = "precomputed"
        X = distance_matrix
    else:
        metric = "euclidean"
        X = embeddings_reduced

    print(f"[INFO] Running HDBSCAN clustering (min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric
    )
    clusters = clusterer.fit_predict(X)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"[INFO] HDBSCAN found {n_clusters} clusters (+ noise)")
    return clusters, embeddings_reduced




# -----------------------------
# Cluster Keyword Analysis (distinctive terms)
# -----------------------------
from keybert import KeyBERT

def cluster_keyphrases(df, clusters, top_n=10, stop_phrases=None, model_name="all-MiniLM-L6-v2"):
    """
    Extract the top distinctive keyphrases for each cluster using KeyBERT.
    stop_phrases: list of phrases to ignore (e.g., repetitive boilerplate)
    """
    if stop_phrases is None:
        stop_phrases = ["just checking", "hello", "hi", "thank you"]

    df = df.copy()
    df['cluster'] = clusters
    kw_model = KeyBERT(model_name)
    keyphrases_per_cluster = {}

    for cluster_id in sorted(df['cluster'].unique()):
        idx = df[df['cluster'] == cluster_id].index
        if len(idx) == 0:
            continue

        cluster_texts = df.loc[idx, "full_text"].tolist()
        cluster_text = " ".join(cluster_texts)

        # Extract keyphrases
        keyphrases = kw_model.extract_keywords(
            cluster_text, 
            keyphrase_ngram_range=(1,3), 
            stop_words="english", 
            top_n=top_n
        )

        # Filter stop phrases
        keyphrases_filtered = [(phrase, score) for phrase, score in keyphrases
                               if phrase.lower() not in stop_phrases]

        keyphrases_df = pd.DataFrame(keyphrases_filtered, columns=["term", "score"])
        keyphrases_per_cluster[cluster_id] = keyphrases_df

    return keyphrases_per_cluster


# -----------------------------
# Cluster visualization
# -----------------------------
def plot_clusters(embeddings, clusters, output_file="clusters.png"):
    n_samples = embeddings.shape[0]
    n_neighbors = min(30, n_samples - 1)
    reducer = UMAP(n_components=5, random_state=42, n_neighbors=n_neighbors)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10,6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], c=clusters, cmap="viridis", alpha=0.6)
    plt.title("HDBSCAN Clustering of Calls")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(output_file)
    plt.close()
    print(f"[OK] Cluster plot saved to {output_file}")

# -----------------------------
# Outlier Detection
# -----------------------------
def detect_outliers(embeddings, contamination=0.05):
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(embeddings)
    return preds  # -1 = outlier, 1 = normal

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def baseline_tfidf_kmeans(df, n_clusters=5, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf = vectorizer.fit_transform(df["full_text"])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf)

    sil = silhouette_score(tfidf, clusters)
    db = davies_bouldin_score(tfidf.toarray(), clusters)

    print(f"[BASELINE] TF-IDF+KMeans | Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}")
    return clusters, sil, db


def evaluate_clustering(embeddings, clusters):
    mask = clusters != -1  # ignore noise points for HDBSCAN
    if np.sum(mask) < 2:
        return None, None

    sil = silhouette_score(embeddings[mask], clusters[mask])
    db = davies_bouldin_score(embeddings[mask], clusters[mask])

    print(f"[SBERT+HDBSCAN] Silhouette: {sil:.3f} | Davies-Bouldin: {db:.3f}")
    return sil, db

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_topic_coherence(df, clusters, top_n=10):
    df = df.copy()
    df['cluster'] = clusters

    # Build topics as top words per cluster
    topics = []
    for cid in np.unique(clusters):
        if cid == -1:  # skip noise cluster
            continue
        cluster_texts = df[df['cluster']==cid]['full_text'].tolist()
        if len(cluster_texts) == 0:
            continue

        # TF-IDF within the cluster
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform(cluster_texts)
        scores = np.asarray(tfidf.mean(axis=0)).flatten()
        terms = np.array(vectorizer.get_feature_names_out())
        top_terms = terms[scores.argsort()[::-1][:top_n]]
        topics.append(list(top_terms))

    # Prepare texts for coherence
    tokenized_texts = [t.split() for t in df["full_text"].tolist()]
    dictionary = Dictionary(tokenized_texts)

    cm = CoherenceModel(topics=topics, texts=tokenized_texts,
                        dictionary=dictionary, coherence="c_v")
    coherence = cm.get_coherence()

    print(f"[INFO] Topic coherence: {coherence:.3f}")
    return coherence

import seaborn as sns

def report_cluster_distribution(df, clusters, output_file="cluster_sizes.png"):
    df['cluster'] = clusters
    counts = df['cluster'].value_counts().sort_index()
    print("\n[INFO] Cluster size distribution:")
    print(counts)

    plt.figure(figsize=(8,5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Calls")
    plt.title("Cluster Size Distribution")
    plt.savefig(output_file)
    plt.close()
    print(f"[OK] Cluster size plot saved to {output_file}")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def evaluate_with_ground_truth(true_labels, predicted_clusters):
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
    print(f"[GT EVAL] ARI: {ari:.3f} | NMI: {nmi:.3f}")
    return ari, nmi
# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    path_911 = "./911.csv"
    path_meta = "./metadata.csv"

    df = load_and_prepare_data(path_911, path_meta)
    print(f"[INFO] Total rows: {len(df)}")
    print(df["language_detected"].value_counts())

    embeddings = compute_embeddings(df)
    clusters, embeddings_reduced = cluster_embeddings(embeddings)
    df['cluster'] = clusters
    print("[INFO] Clustering done")

    keywords = cluster_keyphrases(df, clusters)
    for cid, df_terms in keywords.items():
        print(f"\nCluster {cid} top terms:\n", df_terms)

    plot_clusters(embeddings_reduced, clusters, output_file="clusters_en.png")

    outliers = detect_outliers(embeddings)
    df['outlier'] = outliers
    n_outliers = np.sum(outliers==-1)
    print(f"[INFO] Detected {n_outliers} potential outlier calls")

    df.to_csv("./out/hoax_clusters_v2.csv", index=False)
    print("[OK] Enriched dataset saved with clusters, keywords, and embeddings")

    evaluate_clustering(embeddings, clusters)
    print("[INFO] Running baseline TF-IDF + KMeans")
    baseline_tfidf_kmeans(df, n_clusters=len(set(clusters))-1)
    print("[INFO] Computing topic coherence")
    compute_topic_coherence(df, clusters, top_n=10)
    print("[INFO] Reporting cluster distribution")
    report_cluster_distribution(df, clusters, output_file="cluster_sizes_en_new.png")
    print("[OK] Pipeline completed")
    

if __name__ == "__main__":
    main()
