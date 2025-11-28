import json
import numpy as np
import pandas as pd
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    print("Loading your saved posts...")
    try:
        with open('raw_data.json', 'r', encoding='utf-8') as f:
            raw_data = json.loads(f.read())
    except FileNotFoundError:
        print("Error: 'raw_data.json' not found.")
        return

    # 1. Prepare Data
    df = pd.DataFrame(raw_data)
    df['caption'] = df['caption'].fillna('')
    # Relaxed filter: Keep posts even with short captions so we don't lose data
    df = df.reset_index(drop=True)
    
    if df.empty:
        print("No data found.")
        return

    print(f"Processing {len(df)} posts...")

    # 2. Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['caption'].tolist(), show_progress_bar=True)

    # 3. Projection
    print("Projecting into 3D space...")
    # Perplexity lowered to 15 to handle smaller datasets better if needed
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(df)-1)) 
    projection = tsne.fit_transform(embeddings)

    # 4. Clustering
    # INCREASED CLUSTERS: 45 categories allows for much finer grouping (e.g. separating "Dogs" from "Cats")
    num_clusters = 45
    print(f"Assigning colors for {num_clusters} categories...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(projection)
    
    df['cluster'] = labels
    df['x'], df['y'], df['z'] = projection[:, 0], projection[:, 1], projection[:, 2]

    # Normalize
    for col in ['x', 'y', 'z']:
        df[col] = (df[col] - df[col].mean()) / df[col].std() * 100

    # 5. Smart Labeling (Robust Version)
    print("Generating smart labels...")
    
    # Extended Stopwords List
    insta_stop_words = list(set([
        'the','and','is','in','to','for','of','with','on','at','from','by','a','an','this','that','it','my','your','we','are','so','but','not',
        'link','bio','post','reels','instagram','video','like','follow','share','save', 'click', 'check', 'more', 'daily', 'page', 'dm', 'comment', 
        'below', 'credit', 'reserved', 'double', 'tap', 'tag', 'friend', 'new', 'update', 'via', 'shot', 'photo', 'picture'
    ]))

    cluster_labels_map = {}
    
    # Prepare text for TF-IDF
    cluster_texts = []
    active_cluster_ids = []
    
    for group_id in range(num_clusters):
        group_df = df[df['cluster'] == group_id]
        if not group_df.empty:
            text = " ".join(group_df['caption'].astype(str).tolist()).lower()
            text = re.sub(r'[^a-z\s]', '', text) # Keep only letters
            cluster_texts.append(text)
            active_cluster_ids.append(group_id)

    try:
        # Try TF-IDF (The Smart Way)
        if not any(len(t.strip()) > 0 for t in cluster_texts):
             raise ValueError("Text too sparse for TF-IDF")

        # min_df=1 ensures we don't crash if words are rare
        tfidf = TfidfVectorizer(stop_words=insta_stop_words, max_df=0.8, min_df=1) 
        tfidf_matrix = tfidf.fit_transform(cluster_texts)
        feature_names = np.array(tfidf.get_feature_names_out())
        
        for idx, group_id in enumerate(active_cluster_ids):
            row = tfidf_matrix[idx].toarray().flatten()
            if row.sum() == 0:
                 cluster_labels_map[group_id] = f"Group {group_id}"
                 continue
            
            # Top 2 words
            top_indices = row.argsort()[-2:][::-1]
            top_words = feature_names[top_indices]
            
            valid_words = [w.capitalize() for w in top_words if len(w) > 2]
            cluster_labels_map[group_id] = ", ".join(valid_words) if valid_words else f"Group {group_id}"

    except Exception as e:
        print(f"⚠️ Smart labeling failed ({e}). using fallback method.")
        # Fallback: Simple Frequency Count
        for group_id in range(num_clusters):
             group_df = df[df['cluster'] == group_id]
             if group_df.empty: continue
             text = " ".join(group_df['caption'].astype(str).tolist()).lower()
             words = re.sub(r'[^a-z\s]', '', text).split()
             words = [w for w in words if w not in insta_stop_words and len(w) > 2]
             common = Counter(words).most_common(2)
             label = ", ".join([c[0].capitalize() for c in common]) if common else f"Group {group_id}"
             cluster_labels_map[group_id] = label

    # 6. Build Final Data
    nodes = []
    links = []
    groups_data = {}

    for group_id in range(num_clusters):
        group_df = df[df['cluster'] == group_id]
        if group_df.empty: continue
            
        # Centers
        cx = float(group_df['x'].mean())
        cy = float(group_df['y'].mean())
        cz = float(group_df['z'].mean())
        
        lbl = cluster_labels_map.get(group_id, f"Group {group_id}")
        groups_data[group_id] = { "label": lbl, "center": {"x": cx, "y": cy, "z": cz} }

        # Hub Node
        hub_id = f"hub_{group_id}"
        nodes.append({
            "id": hub_id, "group": group_id, "type": "center", "name": lbl,
            "fx": cx, "fy": cy, "fz": cz
        })

        # Post Nodes
        for _, row in group_df.iterrows():
            pid = row.get('id', row.name)
            nodes.append({
                "id": pid,
                "user": row.get('username', 'unknown'),
                "caption": row['caption'][:100] + "...",
                "img": row.get('image_url', ''),
                "link": row.get('link', '#'),
                "group": int(group_id),
                "type": "post",
                "fx": float(row['x']), "fy": float(row['y']), "fz": float(row['z'])
            })
            links.append({ "source": pid, "target": hub_id })

    # Export
    js_content = f"window.GALAXY_DATA = {json.dumps({'nodes': nodes, 'links': links}, indent=2)};\n"
    js_content += f"window.GROUP_LABELS = {json.dumps(groups_data, indent=2)};"
    
    with open('big_galaxy_data.js', 'w', encoding='utf-8') as f:
        f.write(js_content)

    print(f"Done! Generated 'galaxy_data.js' with {num_clusters} precise categories.")

if __name__ == "__main__":
    main()