import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')
df.head()

# Visualize the data
plt.figure(figsize=(15,10))
sns.countplot(data=df, x='Category')
plt.xticks(rotation=90)
plt.show()

# Define text cleaning function
def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'\bRT\b|\bcc\b', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7F]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# Apply text cleaning
df['Resume'] = df['Resume'].apply(lambda x: clean_resume(x))

# Encode the categories
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Define a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', OneVsRestClassifier(KNeighborsClassifier()))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['Resume'], df['Category'], test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Save the pipeline
with open('model_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Optionally, load the pipeline (if you need to reuse it later)
# with open('model_pipeline.pkl', 'rb') as file:
#     pipeline = pickle.load(file)
