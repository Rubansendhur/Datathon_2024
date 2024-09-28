# Import necessary libraries and packages
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import scipy.sparse as sp
from sklearn.model_selection import RandomizedSearchCV

# Download NLTK data (only needs to be done once)
nltk.download('punkt')  # For tokenization
#nltk.download('stopwords')  # For stopwords

# Load the dataset
data = pd.read_csv("C:\\Users\\ruban\\Downloads\\Login\\Train.csv")
test_data = pd.read_csv("C:\\Users\\ruban\\Downloads\\Login\\Test.csv")

# Preprocess TEXT data
data['ENTITY_DESCRIPTION'] = data['ENTITY_DESCRIPTION'].str.lower()
data['ENTITY_DESCRIPTION'] = data['ENTITY_DESCRIPTION'].str.replace('[^\w\s]', '', regex=True)
data['tokens'] = data['ENTITY_DESCRIPTION'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, min_df=5)
X_text = tfidf.fit_transform(data['ENTITY_DESCRIPTION'])

# Feature Engineering
data['description_length'] = data['ENTITY_DESCRIPTION'].apply(len)
data['word_count'] = data['ENTITY_DESCRIPTION'].apply(lambda x: len(x.split()))
data['avg_word_length'] = data['description_length'] / (data['word_count'] + 1)

# Target Encoding for CATEGORY_ID
mean_lengths = data.groupby('CATEGORY_ID')['ENTITY_LENGTH'].mean()
data['CATEGORY_ID_encoded'] = data['CATEGORY_ID'].map(mean_lengths)

# Scaling numerical features
scaler = StandardScaler()
num_features = ['description_length', 'word_count', 'avg_word_length', 'CATEGORY_ID_encoded']
data[num_features] = scaler.fit_transform(data[num_features])

# Combine TF-IDF matrix with numerical features
X_combined_train = sp.hstack([X_text, data[['description_length', 'word_count', 'avg_word_length', 'CATEGORY_ID_encoded']].values])

# Define target and split data
y = data['ENTITY_LENGTH']  # Use the raw target variable for now
X_train, X_val, y_train, y_val = train_test_split(X_combined_train, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestRegressor(random_state=42)
rf_search = RandomizedSearchCV(rf_model, param_distributions=param_grid_rf, n_iter=10, cv=3, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf_model = rf_search.best_estimator_

# Predictions and Evaluation
y_pred_rf = best_rf_model.predict(X_val)
mape_rf = mean_absolute_percentage_error(y_val, y_pred_rf)
print(f"Optimized Random Forest MAPE: {mape_rf}")

# Apply preprocessing to test data
test_data['ENTITY_DESCRIPTION'] = test_data['ENTITY_DESCRIPTION'].str.lower()
test_data['ENTITY_DESCRIPTION'] = test_data['ENTITY_DESCRIPTION'].str.replace('[^\w\s]', '', regex=True)
test_data['description_length'] = test_data['ENTITY_DESCRIPTION'].apply(len)
test_data['word_count'] = test_data['ENTITY_DESCRIPTION'].apply(lambda x: len(x.split()))
test_data['avg_word_length'] = test_data['description_length'] / (test_data['word_count'] + 1)
test_data['CATEGORY_ID_encoded'] = test_data['CATEGORY_ID'].map(mean_lengths)
test_data[num_features] = scaler.transform(test_data[num_features])

# Combine TF-IDF matrix with numerical features for test set
X_test_tfidf = tfidf.transform(test_data['ENTITY_DESCRIPTION'])
X_test_combined = sp.hstack([X_test_tfidf, test_data[['description_length', 'word_count', 'avg_word_length', 'CATEGORY_ID_encoded']].values])

# Generate predictions on the test set using optimized Random Forest
y_test_pred_rf = best_rf_model.predict(X_test_combined)

# Create submission file
submission_rf = pd.DataFrame({'ENTITY_ID': test_data['ENTITY_ID'], 'ENTITY_LENGTH': y_test_pred_rf})
submission_rf.to_csv('submission_rf_optimized.csv', index=False)
print("Optimized Random Forest submission file created successfully.")

