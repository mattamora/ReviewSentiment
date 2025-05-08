"""
the main file.
Im basically done all i need to do is the slides and report.
"""

# upgrade of your part4.py with logistic regression 
# main model for my project

import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('restaurant_reviews_filtered.csv')

# Relabel sentiment: 3.5 and above is Positive, 3.0 and below is Negative
def label_sentiment(stars):
    return 'Positive' if stars >= 3.5 else 'Negative'

df['nb_sentiment'] = df['stars'].apply(label_sentiment)
df = df[df['nb_sentiment'].isin(['Positive', 'Negative'])]

# Step 1: Lemmatize and clean
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

df['cleaned_text'] = df['text'].apply(clean_and_lemmatize)

# Step 2: TF-IDF with n-grams (unigrams + bigrams)
# tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,         # Remove words that appear in fewer than 2 reviews
    max_df=0.85       # Remove words that appear in more than 85% of reviews
)


X = tfidf.fit_transform(df['cleaned_text'])
y = df['nb_sentiment']

# Train/test split
# 80% training 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nLogistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred))




"""" 
# Test the model on a new review
custom_review = "Food was decent, however it took too long and the service was terrible"

# Preprocess and transform
cleaned = clean_and_lemmatize(custom_review)
X_custom = tfidf.transform([cleaned])

# Predict
prediction = model.predict(X_custom)[0]
print(f"\nSentiment Prediction for custom review:\n\"{custom_review}\"\n→ {prediction}\n")
"""





"""
# Test the model on multiple custom reviews
test_reviews = [
    "The service was excellent and the food came out hot and fresh. I'd definitely come back!",  # Clearly Positive
    "Waited over 30 minutes for cold food and the staff didn’t seem to care at all.",            # Clearly Negative
    "The pizza was decent, though the crust was a little too chewy for my liking.",             # Mild Positive
    "The appetizers were amazing, but the main course was undercooked and disappointing.",       # Mixed sentiment
    "Not the worst place I’ve eaten, but definitely not worth the price or the hype."            # Subtle Negative
]

print("\n Sentiment Predictions for Test Reviews:\n")

for review in test_reviews:
    cleaned = clean_and_lemmatize(review)
    X_custom = tfidf.transform([cleaned])
    prediction = model.predict(X_custom)[0]
    print(f"Review: \"{review}\"\n→ Predicted Sentiment: {prediction}\n")
""" 


""" 
from sklearn.model_selection import cross_val_score

# Use 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("\nCross-Validation Results:")
print("Accuracy scores per fold:", cv_scores)
print("Average cross-validated accuracy: {:.2f}".format(cv_scores.mean()))
"""


""" 
# Show misclassified examples
print("\nMisclassified Reviews:\n")

# Get the indexes where predictions don't match actual labels
misclassified_indexes = (y_test != y_pred)

# Create a DataFrame to display the misclassified samples
misclassified_df = pd.DataFrame({
    'Index': y_test.index[misclassified_indexes],
    'Review': df.loc[y_test.index[misclassified_indexes], 'text'],
    'True Label': y_test[misclassified_indexes].values,
    'Predicted Label': y_pred[misclassified_indexes]
})

# Show top 5 for review
pd.set_option('display.max_colwidth', None) # shows the entire review
# print(misclassified_df.head(10))
# Print each misclassified review in your custom format
for idx, row in misclassified_df.head(3).iterrows():
    print(f"Number: {row['Index']}")
    print(f"Review True Label: {row['True Label']}")
    print(f"Predicted Label: {row['Predicted Label']}")
    print(f"Review: {row['Review']}\n")
"""
