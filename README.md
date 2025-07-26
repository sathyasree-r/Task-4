# Task-4
# 📧 Spam Email Detection using Machine Learning – CodTech Internship Task 4

This project uses a supervised machine learning model to classify email/sms messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) and Scikit-learn.

---

## 📌 Features

✅ Loads real-world dataset (`spam.csv`)  
✅ Preprocesses and vectorizes text using TF-IDF  
✅ Trains a **Naive Bayes** classifier  
✅ Evaluates accuracy, confusion matrix & classification report  
✅ Predicts new custom messages as spam or ham  
✅ Visualizes confusion matrix with `seaborn`

---

## 🛠️ Tech Stack

**Python 3.x**

### Libraries Used:
- `pandas` → Data handling  
- `scikit-learn` → Model building and evaluation  
- `seaborn` / `matplotlib` → Visualization  
- `numpy` → Numerical operations  

---

## 🚀 Installation & Setup

### 1️⃣ Clone this Repository
```bash
git clone https://github.com/your-username/spam-detection-ml.git
cd spam-detection-ml
2️⃣ Install Dependencies
bash
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn
3️⃣ Prepare Your Data
Make sure you have a spam.csv file with this format:

label	text
ham	I'm going to be home soon.
spam	You won a free ticket! Click now!

The script expects columns: label (ham/spam) and text (message)

▶️ Running the Script
Run the model from terminal:

bash
Copy
Edit
python spam-detect.py
✅ Sample Output
sql
Copy
Edit
✅ Dataset Loaded Successfully!
✅ Accuracy: 0.98

🔮 Predictions on Sample Messages:
Congratulations! You won a free lottery ticket, claim now! --> Spam
Hi, are we still meeting tomorrow? --> Ham
📊 Output Visualization
The script generates a confusion matrix heatmap like this:

markdown
Copy
Edit
                 Predicted
               ┌───────┬───────┐
      Actual   │ Ham   │ Spam  │
        Ham    │  965  │   2   │
        Spam   │   8   │  140  │
📂 Project Structure
bash
Copy
Edit
spam-detection-ml/
│
├── spam.csv              # Dataset
├── spam-detect.py        # Python script
├── spam-detect.ipynb     # Optional Jupyter Notebook version
├── README.md             # Project documentation
📋 Task Requirement
This project fulfills Task 4 of the CodTech Internship:

Create a predictive model using Scikit-learn to classify or predict outcomes from a dataset (e.g., spam email detection)
Deliverable: A Jupyter Notebook or Python Script showcasing the model’s implementation and evaluation.

👩‍💻 Author
Sathyasree
🌐 GitHub: github.com/sathyasree-r

🤝 Contributing
Pull requests and suggestions are welcome!

📜 License
This project is licensed under the MIT License.
