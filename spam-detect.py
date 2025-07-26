# 📌 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📌 2. Load Dataset (Make sure spam.csv is in the same folder)
df = pd.read_csv("spam.csv")

print("✅ Dataset Loaded Successfully!")
print(df.head())

# 📌 3. Encode Labels (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham':0, 'spam':1})

# 📌 4. Split Data into Train & Test
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 5. Convert Text to Numerical Form (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 📌 6. Build & Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 📌 7. Make Predictions
y_pred = model.predict(X_test_tfidf)

# 📌 8. Evaluate Model
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📌 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 📌 10. Test on New Messages
sample = ["Congratulations! You won a free lottery ticket, claim now!", 
          "Hi, are we still meeting tomorrow?"]
sample_tfidf = vectorizer.transform(sample)
pred = model.predict(sample_tfidf)
print("\n🔮 Predictions on Sample Messages:")
for msg, label in zip(sample, pred):
    print(f"{msg} --> {'Spam' if label==1 else 'Ham'}")
