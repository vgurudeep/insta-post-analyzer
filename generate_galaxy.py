import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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
    
    # Filter out empty posts
    df['caption'] = df['caption'].fillna('')
    df = df[df['caption'].str.len() > 2].reset_index(drop=True)
    
    print(f"Processing {len(df)} posts. This involves heavy AI math, please wait...")

    # 2. Convert Captions to "Meaning Vectors" (Embeddings)
    # We use a small, fast model suitable for running locally
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['caption'].tolist(), show_progress_bar=True)

    # 3. Reduce Dimensions to 3D (X, Y, Z coordinates)
    # The AI gives us 384 dimensions. We need 3 to see it.
    print("Projecting into 3D space...")
    
    # FIX: Removed 'n_iter=1000' to prevent version conflicts (defaults to 1000 anyway)
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    
    # We reduce to 3 dimensions for x, y, z
    projection = tsne.fit_transform(embeddings)

    # 4. Color Clustering
    # Group them so we can color-code the "stars"
    num_clusters = 15
    print("Assigning colors...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(projection)

    # 5. Build the Galaxy Data Structure
    nodes = []
    
    # We normalized the coordinates to make the graph look centered
    x_norm = (projection[:, 0] - projection[:, 0].mean()) / projection[:, 0].std()
    y_norm = (projection[:, 1] - projection[:, 1].mean()) / projection[:, 1].std()
    z_norm = (projection[:, 2] - projection[:, 2].mean()) / projection[:, 2].std()

    for i, row in df.iterrows():
        nodes.append({
            "id": row.get('id', i),
            "user": row.get('username', 'unknown'),
            "caption": row['caption'][:200] + "...", # Truncate for display
            "img": row.get('image_url', ''),
            "link": row.get('link', '#'),
            "group": int(labels[i]), # Color group
            "fx": float(x_norm[i] * 100), # Spread them out by multiplying
            "fy": float(y_norm[i] * 100),
            "fz": float(z_norm[i] * 100)
        })

    # Export to JS
    js_content = f"window.GALAXY_DATA = {json.dumps({'nodes': nodes, 'links': []}, indent=2)};"
    
    with open('galaxy_data.js', 'w', encoding='utf-8') as f:
        f.write(js_content)

    print("Done! Generated 'galaxy_data.js'. Open 'galaxy.html' to enter the universe.")

if __name__ == "__main__":
    main()