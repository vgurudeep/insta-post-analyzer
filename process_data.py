import json
import re

# --- CONFIGURATION: Define your categories and keywords here ---
CATEGORIES = {
    "Food & Recipes": ["recipe", "food", "cook", "bake", "delicious", "yummy", "dinner", "lunch", "breakfast", "eat", "chef", "ingredients"],
    "Fashion & Style": ["fashion", "style", "outfit", "wear", "clothes", "dress", "shoes", "ootd", "streetwear", "model", "brand"],
    "Tech & AI": ["tech", "ai", "code", "programming", "python", "javascript", "developer", "software", "app", "web", "computer", "robot"],
    "Education & Learning": ["learn", "study", "tip", "guide", "tutorial", "howto", "hack", "trick", "university", "school", "education", "history", "science"],
    "Politics & News": ["politics", "news", "government", "law", "rights", "vote", "policy", "economy", "war", "crisis", "president"],
    "Humour & Memes": ["meme", "funny", "lol", "humor", "comedy", "joke", "laugh", "prank", "satire"],
    "Fitness & Health": ["fitness", "gym", "workout", "health", "diet", "exercise", "muscle", "run", "yoga", "sport"],
    "Art & Design": ["art", "design", "illustration", "drawing", "sketch", "creative", "artist", "painting", "decor", "interior"],
    "Travel": ["travel", "trip", "vacation", "holiday", "view", "nature", "beach", "mountain", "explore", "city"]
}

def clean_caption(text):
    if not text:
        return ""
    return text.lower()

def categorize_post(caption):
    caption = clean_caption(caption)
    scores = {cat: 0 for cat in CATEGORIES}

    # Simple keyword scoring
    for category, keywords in CATEGORIES.items():
        for word in keywords:
            # Check for whole words or hashtags
            if re.search(r'\b' + re.escape(word) + r'\b', caption) or f"#{word}" in caption:
                scores[category] += 1
    
    # Get the category with the highest score
    best_category = max(scores, key=scores.get)
    
    # Only assign if score > 0, otherwise "Uncategorized"
    if scores[best_category] > 0:
        return best_category
    return "Uncategorized"

def main():
    try:
        # Read the raw JSON file
        with open('raw_data.json', 'r', encoding='utf-8') as f:
            # Handle potential copy-paste artifacts if user copied the console log directly
            content = f.read()
            # If the user copied the console output, it might be wrapped in quotes or have extra text.
            # We assume it is valid JSON for now.
            posts = json.loads(content)
            
        print(f"Loaded {len(posts)} posts. Processing...")

        processed_data = []

        for post in posts:
            # Categorize
            category = categorize_post(post.get('caption', ''))
            
            # Create clean object
            processed_data.append({
                "id": post.get('id'),
                "link": post.get('link'),
                "image": post.get('image_url'),
                "username": post.get('username'),
                "category": category,
                "caption": post.get('caption', '')[:150] + "..." if post.get('caption') else "", # Truncate caption for UI
                "date": post.get('timestamp')
            })

        # Save as a JavaScript file (window.IG_DATA = [...])
        # This allows the HTML file to load it locally without CORS errors
        js_content = f"window.IG_DATA = {json.dumps(processed_data, indent=2)};"
        
        with open('data.js', 'w', encoding='utf-8') as f:
            f.write(js_content)

        print("Success! Created 'data.js'. You can now open 'index.html'.")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your 'raw_data.json' contains valid JSON text.")

if __name__ == "__main__":
    main()