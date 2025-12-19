# Heart Disease Prediction using Machine Learning

This project focuses on building an end-to-end machine learning pipeline using the UCI Heart Disease dataset to analyze health indicators and predict the presence of heart disease.

## Dataset
- Source: UCI Heart Disease Dataset
- Records: 920
- Features: 16 clinical and demographic attributes
- Target:
  - `num` (used for regression)
  - Binary classification (`num > 0`) for disease presence

## Objectives
- Predict the presence of heart disease
- Perform exploratory data analysis (EDA) to identify important health indicators
- Compare multiple regression and classification models
- Apply unsupervised learning (K-Means) for patient clustering

## Methodology
1. **Data Cleaning**
   - Median imputation for numerical features
   - Mode imputation for categorical features
   - Encoding of categorical variables

2. **Exploratory Data Analysis**
   - Distribution analysis
   - Relationship analysis (Age vs Max Heart Rate)
   - Visualization of key patterns

3. **Model Development**
   - Regression: Linear Regression
   - Classification: Logistic Regression, KNN, Decision Tree, Naive Bayes, SVM
   - Unsupervised Learning: K-Means Clustering

4. **Model Evaluation**
   - Regression Metrics: MAE, RMSE, R²
   - Classification Metrics: Accuracy, F1-score, Confusion Matrix, ROC-AUC

## Results
- Classification accuracy achieved around **80–82%**, which is realistic for healthcare datasets
- Linear regression achieved an R² score of approximately **0.47**
- K-Means clustering helped explore patient segmentation patterns

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn

## Key Learnings
- Importance of proper data preprocessing in medical datasets
- Trade-offs between different machine learning models
- Value of realistic evaluation metrics over inflated accuracy

## Future Improvements
- Hyperparameter tuning
- Feature importance analysis
- Cross-validation
- ROC curve visualization

## Author
- Shubham Bhadula
