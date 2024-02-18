import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from "food_data.csv"
data = pd.read_csv("food_data.csv")

# Data Cleaning and Preprocessing
data.dropna(subset=['Item', 'Categories'], inplace=True)
data['Categories'] = data['Categories'].apply(literal_eval)

# Check unique values in 'Categories' column
unique_categories = data['Categories'].explode().unique()
#print(f"Unique Categories: {unique_categories}")

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extract features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train = tfidf_vectorizer.fit_transform(train_data['Item'])
X_test = tfidf_vectorizer.transform(test_data['Item'])

# Transform categories into binary labels
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_data['Categories'])
y_test = mlb.transform(test_data['Categories'])

# Train a RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Example prediction for a new item
new_item_text_keywords = input("Enter the food name: ")

# Extract individual keywords
keywords = new_item_text_keywords.lower().split()

# Initialize lists to store predicted categories for each keyword
predicted_categories_keywords = []

for keyword in keywords:
    # Transform individual keyword using TF-IDF
    keyword_features = tfidf_vectorizer.transform([keyword])

    # Compute cosine similarity between the keyword and items in the dataset
    keyword_similarities = cosine_similarity(keyword_features, X_train)

    # Find the index of the most similar item in the dataset
    most_similar_index_keyword = np.argmax(keyword_similarities)

    # Get the corresponding categories from 'y_train'
    predicted_categories_keyword = mlb.inverse_transform(y_train[most_similar_index_keyword].reshape(1, -1))

    # Append the predicted categories for the keyword
    predicted_categories_keywords.append(predicted_categories_keyword)

# Combine the predicted categories for each keyword
combined_categories = tuple(set(category for categories_keyword in predicted_categories_keywords for category in categories_keyword))
#print(f"Predicted Categories for '{new_item_text_keywords}': {combined_categories}")

category_set=set()
for i in combined_categories:
    for j in i:
        category_set.add(j)
if "Chicken" in category_set: 
    if "Vegetarian" in category_set:
        category_set.remove("Vegetarian")
if "Seafood" in category_set: 
    if "Vegetarian" in category_set:
        category_set.remove("Vegetarian")
if "Non-Veg" in category_set: 
    if "Vegetarian" in category_set:
        category_set.remove("Vegetarian")


print(f"Predicted Categories for '{new_item_text_keywords}': {category_set}")

