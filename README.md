# Customer Churn Prediction – Machine Learning Project

## 1. Project Overview

This project develops a machine-learning-based Customer Churn Prediction System for a telecommunications company.

The objective is to predict whether a customer is likely to leave the service using customer demographic information, subscribed services, contract details and billing information.

The project compares five classification algorithms and provides an interactive Streamlit application for model evaluation and individual customer predictions.

## 2. Dataset Description

**Dataset:** Telco Customer Churn

- Total records: 7,043
- Total original columns: 33
- Target variable: `Churn Value`
- No churn (0): 5,174 customers
- Churn (1): 1,869 customers

The model uses 19 customer features, including tenure, internet service, contract type, payment method, monthly charges and total charges.

The dataset was cleaned and prepared before model training. Missing numerical and categorical values were handled through preprocessing.

## 3. Machine Learning Algorithms

Five classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

The dataset was divided into training and testing sets using an 80:20 split with stratification and random state 42.

Numerical features were imputed and standardized. Categorical features were imputed and one-hot encoded.

## 4. Model Performance Comparison

The following results were obtained on the test dataset.

Model                    Accuracy          AUC          Precision       Recall      F1 Score       MCC
-------------------------------------------------------------------------------------------------------------
Logistic Regression     0.8020           0.8487           0.6435        0.5695       0.6043         0.4744
Decision Tree           0.7353           0.6715           0.5013        0.5294       0.5150         0.3333
KNN                     0.7693           0.7967           0.5636        0.5802       0.5718         0.4141
Naive Bayes             0.7204           0.8150           0.4840        0.8075       0.6052         0.4413
SVM                     0.7970           0.8126           0.6477        0.5160       0.5744         0.4483




## 5. Model Observations

### Logistic Regression
- Achieved the highest accuracy of 0.8020.
- Achieved the highest AUC score of 0.8487.
- Achieved the highest MCC score of 0.4744.
- Provides a strong overall balance of performance.

### Decision Tree
- Achieved an accuracy of 0.7353.
- Had the lowest AUC score among the five models.
- Its overall performance was lower than the other evaluated models.

### K-Nearest Neighbors
- Achieved an accuracy of 0.7693.
- Produced moderate precision, recall and F1 performance.

### Naive Bayes
- Achieved the highest recall of 0.8075.
- Achieved the highest F1 score of 0.6052.
- May be useful when identifying as many potential churn customers as possible is important.

### Support Vector Machine
- Achieved the highest precision of 0.6477.
- Achieved an accuracy of 0.7970.
- Produced competitive overall performance.

## 6. Overall Conclusion

Logistic Regression achieved the highest accuracy, AUC and MCC scores, making it the strongest model under these overall evaluation measures.

Naive Bayes achieved the highest recall and F1 score, making it a useful alternative when detecting more potential churn customers is the priority.

The final model choice depends on the business objective and the relative importance of false positives and false negatives.

## 7. Streamlit Application Features

The deployed application provides:

- CSV test dataset upload.
- Selection of five trained machine-learning models.
- Accuracy, AUC, Precision, Recall, F1 and MCC evaluation metrics.
- Confusion matrix.
- Classification report.
- Individual customer churn prediction.
- Estimated churn probability.

The application uses saved trained models and the saved preprocessing object. It does not retrain models using uploaded test data.

## 8. Project Links

**GitHub Repository:**

https://github.com/sandeepmathurril-blip/Customer-Churn-Prediction

**Live Streamlit Application:**

https://customer-churn-prediction-ta3wz3ceqnbuah6j8k8x4t.streamlit.app

## 9. Project Files

- `Assignment2_MachineLearning.ipynb` – Data analysis, preprocessing, model training and evaluation.
- `app.py` – Streamlit application.
- `model/` – Five saved trained classification models.
- `preprocessor.pkl` – Saved preprocessing object.
- `test_data.csv` – Test dataset for application evaluation.
- `ML_Model_Comparison.csv` – Model comparison results.
- `requirements.txt` – Python dependencies.
- `Telco_customer_churn.csv` – Original dataset.

## 10. How to Run the Application

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload the test CSV in the Model Evaluation section to compare the five trained models.

---

**Project:** Customer Churn Prediction  
**Technology:** Python, Pandas, NumPy, Scikit-learn, Joblib and Streamlit

