# Kaiburr Task 5 — Data Science (Perform a Text Classification on consumer complaint dataset)

## Candidate Details
**Name:** R Devika  
**Date:** 18 October 2025  
**Task:** Data Science example

## Overview
This project performs **Text Classification** on the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) published by the **Consumer Financial Protection Bureau (CFPB)**.  
The goal is to classify consumer complaint narratives into one of the following four categories:

| Label | Category |
|:--|:--|
| 0 | Credit reporting, repair, or other personal consumer reports |
| 1 | Debt collection |
| 2 | Consumer Loan |
| 3 | Mortgage |

---

## Steps Performed

### 1. Data Loading
- The dataset was obtained from the CFPB API CSV endpoint linked on data.gov.  
- Only records with available **consumer complaint narratives** were considered.

<img width="1834" height="1026" alt="image" src="https://github.com/user-attachments/assets/d4e314a4-a18a-44e1-a200-6c3270306e3f" />

### 2. Exploratory Data Analysis (EDA)
- Inspected product categories and complaint distributions.  
- Visualized **class distribution** using Seaborn count plots.  
- Generated a **Word Cloud** to show frequently used terms.
<img width="1842" height="1028" alt="image" src="https://github.com/user-attachments/assets/ab1e4108-2805-4d9d-a8be-d07bd27b593b" />
<img width="1857" height="1029" alt="image" src="https://github.com/user-attachments/assets/e2b15802-5d38-4c06-9dfe-d178fe99f4a3" />


### 3. Text Pre-Processing
- Lower-casing, punctuation removal, and stop-word removal (using `nltk`).  
- Cleaned text stored in a new column `clean_text`.
<img width="1840" height="1030" alt="image" src="https://github.com/user-attachments/assets/7dfc4a94-1b76-48a4-8bdc-3b05bd32761d" />


### 4. Feature Engineering
- Converted text data into numerical features using **TF-IDF Vectorization** (`max_features = 5000`).
```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### 5. Model Building & Comparison
- Evaluated multiple models:
  - Logistic Regression  
  - Multinomial Naive Bayes  
  - Random Forest Classifier  
- Compared accuracy scores and visualized results.
<img width="1826" height="1029" alt="image" src="https://github.com/user-attachments/assets/74a8234b-0f10-4810-9ade-2663cddb9ca2" />


### 6. Model Evaluation
- Generated classification reports (precision, recall, F1).  
- Displayed confusion matrix heatmap for the best performing model.
<img width="1853" height="1026" alt="image" src="https://github.com/user-attachments/assets/4f59713a-e0df-4d03-b508-54e583bad712" />


### 7. Prediction
- Tested the final model on sample complaint texts and displayed predicted categories.
<img width="1840" height="1028" alt="image" src="https://github.com/user-attachments/assets/6482a6fd-7ef5-448e-a2dc-7f304b11e564" />


---

## Libraries Used
- pandas  
- numpy  
- matplotlib  
- seaborn  
- wordcloud  
- nltk  
- scikit-learn  


