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
    df = df[df['caption'].str.len() > 2].reset_index(drop=True)
    
    if df.empty:
        print("No valid text data found to analyze.")
        return

    print(f"Processing {len(df)} posts. This involves heavy AI math, please wait...")

    # 2. Convert Captions to "Meaning Vectors" (Embeddings)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['caption'].tolist(), show_progress_bar=True)

    # 3. Reduce Dimensions to 3D (X, Y, Z coordinates)
    print("Projecting into 3D space...")
    # NOTE: n_iter removed to avoid version conflict
    tsne = TSNE(n_components=3, random_state=42, perplexity=30) 
    projection = tsne.fit_transform(embeddings)

    # 4. Color Clustering
    num_clusters = 15
    print("Assigning colors and centers...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(projection)
    
    df['cluster'] = labels
    df['x'], df['y'], df['z'] = projection[:, 0], projection[:, 1], projection[:, 2]

    # Normalize coordinates to make the galaxy centered and spread out
    df['x'] = (df['x'] - df['x'].mean()) / df['x'].std() * 100
    df['y'] = (df['y'] - df['y'].mean()) / df['y'].std() * 100
    df['z'] = (df['z'] - df['z'].mean()) / df['z'].std() * 100

    # 5. Build Nodes and Links
    nodes = []
    links = []
    
    # --- STRATEGY UPDATE: TF-IDF for Dynamic Labeling ---
    print("Generating smart category labels (TF-IDF)...")
    
    # A. Aggregate text per cluster
    cluster_texts = []
    active_cluster_ids = []
    
    for group_id in range(num_clusters):
        group_df = df[df['cluster'] == group_id]
        if not group_df.empty:
            # Join all captions, simple cleanup
            text = " ".join(group_df['caption'].astype(str).tolist()).lower()
            text = re.sub(r'[^a-z\s]', '', text) 
            cluster_texts.append(text)
            active_cluster_ids.append(group_id)
            
    # B. Define Instagram-specific Stopwords (to ignore)
    insta_stop_words = [
        'the','and','is','in','to','for','of','with','on','at','from','by','a','an','this','that','it','my','your','we','are','so','but','not',
        'link','bio','post','reels','instagram','video','like','follow','share','save', 'click', 'check', 'more', 'daily', 'page', 'dm', 'comment', 'below', 'credit', 'reserved'
    ]
    
    # C. Run TF-IDF
    # max_df=0.6 means: "Ignore words that appear in more than 60% of the clusters"
    # This aggressively removes generic words like "viral" or "trending" if they are everywhere.
    tfidf = TfidfVectorizer(stop_words=insta_stop_words, max_df=0.6) 
    tfidf_matrix = tfidf.fit_transform(cluster_texts)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    cluster_labels_map = {}
    
    for idx, group_id in enumerate(active_cluster_ids):
        # Get the row for this cluster
        row = tfidf_matrix[idx].toarray().flatten()
        # Get indices of top 2 highest scoring words
        top_indices = row.argsort()[-2:][::-1]
        top_words = feature_names[top_indices]
        
        if len(top_words) > 0:
            label = ", ".join([w.capitalize() for w in top_words])
        else:
            label = f"Group {group_id}"
        cluster_labels_map[group_id] = label

    # Store group data for labeling in JS
    groups_data = {}

    for group_id in range(num_clusters):
        group_df = df[df['cluster'] == group_id]
        if group_df.empty:
            continue
            
        # Calculate Group Center (Average Position)
        center_x = float(group_df['x'].mean())
        center_y = float(group_df['y'].mean())
        center_z = float(group_df['z'].mean())
        
        # Get Smart Label
        category_name = cluster_labels_map.get(group_id, f"Group {group_id}")

        groups_data[group_id] = {
            "label": category_name,
            "color": None, 
            "center": {"x": center_x, "y": center_y, "z": center_z}
        }

        # Create Group Center Node (The "Hub")
        group_center_id = f"group_{group_id}"
        nodes.append({
            "id": group_center_id,
            "group": group_id,
            "type": "center",
            "name": category_name,
            "fx": center_x, "fy": center_y, "fz": center_z,
            "size": 30 
        })

        # Create Links from posts to the Center
        for _, row in group_df.iterrows():
            post_id = row.get('id', row.name)
            
            # Add Post Node
            nodes.append({
                "id": post_id,
                "user": row.get('username', 'unknown'),
                "caption": row['caption'][:200] + "...",
                "img": row.get('image_url', ''),
                "link": row.get('link', '#'),
                "group": int(group_id),
                "type": "post",
                "fx": float(row['x']), "fy": float(row['y']), "fz": float(row['z']),
                "size": 5
            })

            # Add Link
            links.append({
                "source": post_id,
                "target": group_center_id,
                "value": 1
            })

    # Export to JS
    js_content = f"window.GALAXY_DATA = {json.dumps({'nodes': nodes, 'links': links}, indent=2)};\n"
    js_content += f"window.GROUP_LABELS = {json.dumps(groups_data, indent=2)};"
    
    with open('new_galaxy_data.js', 'w', encoding='utf-8') as f:
        f.write(js_content)

    print("Done! Generated 'galaxy_data.js' with centers and links.")

if __name__ == "__main__":
    main()