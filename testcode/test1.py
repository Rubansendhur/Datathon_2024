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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import scipy.sparse as sp

# Download NLTK data (only needs to be done once)
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords

# Load the dataset
data = pd.read_csv("C:\\Users\\ruban\\Downloads\\Login\\Train.csv")
test_data = pd.read_csv("C:\\Users\\ruban\\Downloads\\Login\\Test.csv")

# Step 1: Understand the Structure of the Data
print(data.shape)
print(data.head())
print(data.info())

# Step 2: Check for Missing Values
print(data.isnull().sum())

# Step 3: Plot the Distribution of ENTITY_LENGTH before any transformation
plt.figure(figsize=(10, 6))
sns.histplot(data['ENTITY_LENGTH'], bins=30, kde=True)
plt.title('Distribution of Entity Length (Before Transformation)')
plt.show()

# Step 4: Handle Outliers Using IQR Method
Q1 = data['ENTITY_LENGTH'].quantile(0.25)
Q3 = data['ENTITY_LENGTH'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_mask = (data['ENTITY_LENGTH'] < lower_bound) | (data['ENTITY_LENGTH'] > upper_bound)
mean_lengths = data.groupby('CATEGORY_ID')['ENTITY_LENGTH'].mean()
data.loc[outlier_mask, 'ENTITY_LENGTH'] = data.loc[outlier_mask, 'CATEGORY_ID'].map(mean_lengths)

# Step 5: Box-Cox Transformation
data['ENTITY_LENGTH'] = data['ENTITY_LENGTH'].replace(0, 1e-3)
data['length_boxcox'], fitted_lambda = stats.boxcox(data['ENTITY_LENGTH'])
print(f'Lambda used for Box-Cox transformation: {fitted_lambda}')

# Step 6: Visualize the Distribution after Box-Cox Transformation
plt.figure(figsize=(10, 6))
sns.histplot(data['length_boxcox'], bins=30, kde=True)
plt.title('Distribution of Entity Length After Box-Cox Transformation')
plt.show()

# Step 7: Check skewness after transformation
skewness_boxcox = pd.Series(data['length_boxcox']).skew()
print(f'Skewness of length_boxcox: {skewness_boxcox}')

# Step 8: Preprocess TEXT data
data['ENTITY_DESCRIPTION'] = data['ENTITY_DESCRIPTION'].str.lower()
data['ENTITY_DESCRIPTION'] = data['ENTITY_DESCRIPTION'].str.replace('[^\w\s]', '', regex=True)
data['tokens'] = data['ENTITY_DESCRIPTION'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

# Step 9: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=10000, min_df=5)
X_text = tfidf.fit_transform(data['ENTITY_DESCRIPTION'])
print("Shape of TF-IDF matrix:", X_text.shape)

# Step 10: Feature Engineering
data['description_length'] = data['ENTITY_DESCRIPTION'].apply(len)
data['word_count'] = data['ENTITY_DESCRIPTION'].apply(lambda x: len(x.split()))
data['avg_word_length'] = data['description_length'] / (data['word_count'] + 1)

# Step 11: Target Encoding for CATEGORY_ID
mean_lengths = data.groupby('CATEGORY_ID')['ENTITY_LENGTH'].mean()
data['CATEGORY_ID_encoded'] = data['CATEGORY_ID'].map(mean_lengths)

# Step 12: Scaling numerical features
scaler = StandardScaler()
num_features = ['description_length', 'word_count', 'avg_word_length', 'CATEGORY_ID_encoded']
data[num_features] = scaler.fit_transform(data[num_features])

# Step 13: Combine TF-IDF matrix with other features
X_combined_train = sp.hstack([X_text, data[num_features].values])
y = data['length_boxcox']

# Step 14: Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_combined_train, y, test_size=0.2, random_state=42)
print("Train-Validation Split:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Step 15: Model Building and Tuning

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_val)

# Random Forest Model with Hyperparameter Tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestRegressor(random_state=42)
rf_search = RandomizedSearchCV(rf_model, param_distributions=param_grid_rf, n_iter=10, cv=3, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf_model = rf_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_val)

# XGBoost Model
xgboost_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgboost_model.fit(X_train, y_train)
y_pred_xgb = xgboost_model.predict(X_val)

# Step 16: Evaluate Models
y_pred_linear = np.nan_to_num(y_pred_linear, nan=0.0)
y_pred_rf = np.nan_to_num(y_pred_rf, nan=0.0)
y_pred_xgb = np.nan_to_num(y_pred_xgb, nan=0.0)
y_val = np.nan_to_num(y_val, nan=0.0)

mape_linear = mean_absolute_percentage_error(y_val, y_pred_linear)
mape_rf = mean_absolute_percentage_error(y_val, y_pred_rf)
mape_xgb = mean_absolute_percentage_error(y_val, y_pred_xgb)

print(f"MAPE - Linear Regression: {mape_linear}")
print(f"MAPE - Random Forest: {mape_rf}")
print(f"MAPE - XGBoost: {mape_xgb}")

# Step 17: Test Set Processing
test_data['ENTITY_DESCRIPTION'] = test_data['ENTITY_DESCRIPTION'].str.lower()
test_data['ENTITY_DESCRIPTION'] = test_data['ENTITY_DESCRIPTION'].str.replace('[^\w\s]', '', regex=True)
test_data['description_length'] = test_data['ENTITY_DESCRIPTION'].apply(len)
test_data['word_count'] = test_data['ENTITY_DESCRIPTION'].apply(lambda x: len(x.split()))
test_data['avg_word_length'] = test_data['description_length'] / (test_data['word_count'] + 1)
test_data['CATEGORY_ID_encoded'] = test_data['CATEGORY_ID'].map(mean_lengths)
test_data[num_features] = scaler.transform(test_data[num_features])

# Apply TF-IDF transformation to test set
X_test_tfidf = tfidf.transform(test_data['ENTITY_DESCRIPTION'])
X_test_combined = sp.hstack([X_test_tfidf, test_data[num_features].values])

# Make predictions on the test set
y_test_pred_xgb = xgboost_model.predict(X_test_combined)

# Ensure no NaN in predictions
y_test_pred_xgb = np.nan_to_num(y_test_pred_xgb, nan=0.0)

# Step 18: Submission
submission = pd.DataFrame({'ENTITY_ID': test_data['ENTITY_ID'], 'ENTITY_LENGTH': y_test_pred_xgb})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully.")
