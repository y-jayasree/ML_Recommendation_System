import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the combined dataset
combined_df = pd.read_csv('combined_dataset.csv')

# Vectorize the product descriptions and search terms
tfidf = TfidfVectorizer(stop_words='english')
product_desc_tfidf = tfidf.fit_transform(combined_df['product_description'])

# Function to get recommendations based on search term
def get_recommendations(search_term, top_n=5):
    search_term_tfidf = tfidf.transform([search_term])
    similarities = cosine_similarity(search_term_tfidf, product_desc_tfidf).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = combined_df.iloc[top_indices][['product_title', 'relevance', 'product_description']]
    recommendations['similarity'] = similarities[top_indices]
    return recommendations

# Visualization 1: Distribution of Relevance Scores
plt.figure(figsize=(10, 6))
sns.histplot(combined_df['relevance'], bins=10, kde=True)
plt.title('Distribution of Product Relevance Scores')
plt.xlabel('Relevance Score')
plt.ylabel('Frequency')
plt.show()

# Visualization 2: Top 10 Search Terms
top_search_terms = combined_df['search_term'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_search_terms.plot(kind='bar', color='skyblue')
plt.title('Top 10 Search Terms')
plt.xlabel('Search Term')
plt.ylabel('Frequency')
plt.show()

# Visualization 3: Similarity Scores Distribution
search_term = "Emergency lamp"
recommendations = get_recommendations(search_term, top_n=10)
plt.figure(figsize=(10, 6))
sns.histplot(recommendations['similarity'], bins=10, kde=True)
plt.title('Distribution of Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.show()

# Visualization 4: Top Recommended Products
plt.figure(figsize=(10, 6))
sns.barplot(x='similarity', y='product_title', data=recommendations)
plt.title(f'Top Recommendations for Search Term: "{search_term}"')
plt.xlabel('Similarity Score')
plt.ylabel('Product Title')
plt.show()

# Visualization 5: Word Cloud for Product Descriptions
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(combined_df['product_description']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Product Descriptions')
plt.show()

# Visualization 6: TF-IDF Feature Importance
feature_names = tfidf.get_feature_names_out()
tfidf_scores = product_desc_tfidf.sum(axis=0).A1
tfidf_df = pd.DataFrame({'Term': feature_names, 'Score': tfidf_scores})
top_tfidf = tfidf_df.sort_values(by='Score', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Term', data=top_tfidf, palette='viridis')
plt.title('Top 10 TF-IDF Features')
plt.xlabel('TF-IDF Score')
plt.ylabel('Term')
plt.show()

# Visualization 7: Word Cloud for Product Titles
wordcloud_titles = WordCloud(width=800, height=400, background_color='white').generate(' '.join(combined_df['product_title']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_titles, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Product Titles')
plt.show()

# Visualization 8: Search Terms vs. Relevance
combined_df['search_length'] = combined_df['search_term'].str.len()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='search_length', y='relevance', data=combined_df)
plt.title('Search Term Length vs. Relevance')
plt.xlabel('Search Term Length')
plt.ylabel('Relevance Score')
plt.show()

# Visualization 9: Distribution of Product Categories
if 'category' in combined_df.columns:
    plt.figure(figsize=(10, 6))
    combined_df['category'].value_counts().plot(kind='bar', color='orange')
    plt.title('Distribution of Product Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()

# Visualization 10: Relevance vs. Similarity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='similarity', y='relevance', data=recommendations)
plt.title('Relevance Score vs. Similarity Score')
plt.xlabel('Similarity Score')
plt.ylabel('Relevance Score')
plt.show()
search_term = "metal plate cover gcfi"
recommendations = get_recommendations(search_term, top_n=5)
print(recommendations)
