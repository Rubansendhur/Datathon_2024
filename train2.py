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

# Download NLTK data (only needs to be done once)
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords

# Load the dataset
data = pd.read_csv("C:\\Users\\ruban\\Downloads\\Login\\Train.csv")

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

# Step 4: Handle Outliers Using IQR Method (optional if needed)
Q1 = data['ENTITY_LENGTH'].quantile(0.25)
Q3 = data['ENTITY_LENGTH'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_mask = (data['ENTITY_LENGTH'] < lower_bound) | (data['ENTITY_LENGTH'] > upper_bound)
mean_lengths = data.groupby('CATEGORY_ID')['ENTITY_LENGTH'].mean()
data.loc[outlier_mask, 'ENTITY_LENGTH'] = data.loc[outlier_mask, 'CATEGORY_ID'].map(mean_lengths)

# Step 5: Box-Cox Transformation
# Ensure no zero or negative values, adding a small constant (1e-3) to avoid issues
data['ENTITY_LENGTH'] = data['ENTITY_LENGTH'].replace(0, 1e-3)

# Apply Box-Cox transformation
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

# Step 8: Preprocess TEXT data (lowercase, punctuation removal, tokenization, stopword removal, stemming)
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

# Step 10: Feature Engineering: Adding text length features
data['description_length'] = data['ENTITY_DESCRIPTION'].apply(len)
data['word_count'] = data['ENTITY_DESCRIPTION'].apply(lambda x: len(x.split()))
data['avg_word_length'] = data['description_length'] / (data['word_count'] + 1)

# Step 11: Target Encoding for CATEGORY_ID
# Calculate the mean ENTITY_LENGTH for each CATEGORY_ID
mean_lengths = data.groupby('CATEGORY_ID')['ENTITY_LENGTH'].mean()

# Map the mean length back to the original dataframe as encoded CATEGORY_ID
data['CATEGORY_ID_encoded'] = data['CATEGORY_ID'].map(mean_lengths)

# Step 12: Scaling numerical features
scaler = StandardScaler()
num_features = ['description_length', 'word_count', 'avg_word_length', 'CATEGORY_ID_encoded']  # Include encoded CATEGORY_ID
data[num_features] = scaler.fit_transform(data[num_features])

# Step 13: Train-test split
X = data.drop(columns=['ENTITY_LENGTH', 'ENTITY_ID', 'CATEGORY_ID', 'ENTITY_DESCRIPTION', 'tokens'])  # Drop unnecessary columns
y = data['length_boxcox']  # Use the Box-Cox transformed target variable

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-Validation Split:", X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# Step 14: Check for multicollinearity
corr_matrix = data[num_features].corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

# The data is now ready for model building with the normalized 'length_boxcox' as the target variable.
