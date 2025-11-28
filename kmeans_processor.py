import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download('stopwords', quiet=True)

def clean_text(text):
    if not text: return ""
    # Remove special chars, keep only letters/numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return text

def main():
    print("Loading data...")
    try:
        with open('raw_data.json', 'r', encoding='utf-8') as f:
            raw_data = json.loads(f.read())
    except FileNotFoundError:
        print("Error: 'raw_data.json' not found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    # 1. Preprocessing
    print("Cleaning text...")
    stop_words = set(stopwords.words('english'))
    # Add common instagram filler words to stoplist
    stop_words.update(['link', 'bio', 'follow', 'like', 'share', 'comment', 'save', 'post', 'instagram', 'reels', 'video'])
    
    df['clean_caption'] = df['caption'].apply(clean_text)
    
    # Remove rows with empty captions
    df = df[df['clean_caption'].str.len() > 3]

    if df.empty:
        print("No valid text data found to analyze.")
        return

    # 2. Vectorization (Convert text to numbers)
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        max_features=1000,  # Only look at top 1000 words
        max_df=0.9,         # Ignore words that appear in >90% of posts
        min_df=2            # Ignore words that appear in <2 posts
    )
    X = vectorizer.fit_transform(df['clean_caption'])

    # 3. Clustering (K-Means)
    # We attempt to find 12 distinct interest groups. You can change n_clusters.
    num_clusters = 25
    print(f"Clustering into {num_clusters} categories...")
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(X)
    
    df['cluster'] = km.labels_

    # 4. Generate Dynamic Category Names
    print("Naming categories...")
    cluster_names = {}
    
    # For Network Chart: Link categories to keywords
    network_nodes = []
    network_links = []
    
    # Create category nodes first
    for i in range(num_clusters):
        # Get all text in this cluster
        cluster_text = " ".join(df[df['cluster'] == i]['clean_caption'])
        
        # Find top 3 most common meaningful words
        words = [w for w in cluster_text.split() if w not in stop_words and len(w) > 2]
        common = Counter(words).most_common(3)
        
        # Name the cluster (e.g., "food-recipe-cook")
        label = "-".join([w[0] for w in common])
        cluster_names[i] = label
        
        # Add to Network Data
        network_nodes.append({
            "id": f"cat_{i}",
            "name": label.upper(),
            "symbolSize": 30 + (len(df[df['cluster'] == i]) / 5), # Size based on post count
            "category": i,
            "label": {"show": True}
        })

        # Add Keyword nodes and links
        for word, freq in common:
            word_id = f"kw_{word}"
            # Check if node exists, if not add it
            if not any(n['id'] == word_id for n in network_nodes):
                network_nodes.append({
                    "id": word_id,
                    "name": word,
                    "symbolSize": 10 + (freq / 10),
                    "category": i,
                    "itemStyle": { "color": "#a8a8a8" }
                })
            
            network_links.append({
                "source": f"cat_{i}",
                "target": word_id
            })

    # Apply names to dataframe
    df['category_name'] = df['cluster'].map(cluster_names)

    # 5. Prepare Data for Bubble Chart
    # Format: { "name": "root", "children": [ { "name": "Category", "value": count } ] }
    bubble_children = []
    for cat_name, group in df.groupby('category_name'):
        bubble_children.append({
            "name": cat_name,
            "value": len(group),
            "children": [
                # Optional: Add individual posts as tiny bubbles inside (limit to top 10 for performance)
                {"name": row['username'], "value": 1, "link": row.get('link')} 
                for _, row in group.head(10).iterrows()
            ]
        })

    bubble_data = {
        "name": "My Interests",
        "children": bubble_children
    }

    # 6. Export to JS
    output_data = {
        "bubble": bubble_data,
        "network": {
            "nodes": network_nodes,
            "links": network_links,
            "categories": [{"name": v} for k,v in cluster_names.items()]
        },
        "raw_stats": df[['link', 'username', 'category_name', 'image_url']].to_dict(orient='records')
    }

    js_content = f"window.VIZ_DATA = {json.dumps(output_data, indent=2)};"
    
    with open('viz_data.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
        
    print("Done! Generated 'viz_data.js'. Open 'dashboard.html' to see the magic.")

if __name__ == "__main__":
    main()