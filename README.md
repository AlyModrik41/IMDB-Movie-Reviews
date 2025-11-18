# IMDB Movie Reviews Sentiment Analysis (TF-IDF From Scratch)

This project performs sentiment analysis on the IMDB Movie Reviews dataset using a **TF-IDF vectorizer implemented entirely from scratch**, followed by machine learning models (Logistic Regression and SVM).  
It demonstrates how NLP preprocessing, tokenization, TF, IDF, and TF-IDF work behind the scenes — without using scikit-learn's TfidfVectorizer.

---

## 🔧 Project Features
- Full **text preprocessing pipeline**:
  - HTML tag removal  
  - URL removal  
  - Tokenization  
  - Stopword removal  
  - POS-aware lemmatization  
  - Lowercasing  
- **Vocabulary construction**
- **Manual TF-IDF computation**
- Machine learning models:
  - Logistic Regression  
  - Support Vector Machine (SVM)
- Evaluation using multiple metrics

---

## 📂 Dataset
The project uses the IMDB Movie Reviews Dataset (50,000 labeled reviews).

A **train–test split of 80% training and 20% testing** is used.

---

## 🧮 Model Performance

### **1️⃣ Logistic Regression**
**Accuracy:** 0.8752

| Metric | Negative | Positive |
|--------|----------|----------|
| Precision | 0.88 | 0.87 |
| Recall    | 0.86 | 0.89 |
| F1-score  | 0.87 | 0.88 |

**Weighted Avg F1:** 0.88  
**Macro Avg F1:** 0.88

---

### **2️⃣ Support Vector Machine (SVM)**
**Accuracy:** 0.9008

| Metric | Class 0 | Class 1 |
|--------|---------|---------|
| Precision | 0.91 | 0.90 |
| Recall    | 0.89 | 0.91 |
| F1-score  | 0.90 | 0.90 |

**Weighted Avg F1:** 0.90  
**Macro Avg F1:** 0.90

---

## 🧠 Why Build TF-IDF From Scratch?
- Understand the mathematics behind TF and IDF  
- Learn how text is transformed into numerical features  
- Visualize how ML models interpret raw text  
- Improve Python vectorization and algorithm skills  
- Gain deeper NLP intuition

---

## 📌 Technologies Used
- Python  
- NumPy  
- Pandas  
- NLTK (tokenization, stopwords, POS tagging, lemmatization)  
- scikit-learn (models + evaluation only)

---

## 🚀 How to Run
```bash
pip install numpy pandas nltk scikit-learn
python main.py
