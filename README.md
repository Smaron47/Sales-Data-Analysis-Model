# Sales-Data-Analysis-Model
The purpose of this script is to analyze sales data and build predictive models using various machine learning algorithms. 
# Sales Data Analysis and Predictive Modeling Documentation

## Objective

The purpose of this script is to analyze sales data and build predictive models using various machine learning algorithms. The models aim to classify clients based on whether their total sales exceed the median, leveraging multiple functional perspectives such as Customer Engagement, Lead Generation, Stock Management, Forecasting, and Lead Prioritization.

---

## Dataset

* **Input File:** `transformed_sales_data.csv`
* **Target Variable:**

  * `Target` (binary): Indicates whether a client's total sales (`TotalClientSales`) are above the median (1) or not (0).

---

## Feature Sets by Business Functionality

The dataset includes multiple features, grouped by business-related objectives:

1. **Customer Engagement & Retention**

   * ClientCode
   * TotalClientSales
   * ClientOrderCount
   * LastPurchaseDate
   * EngagementScore

2. **Lead Generation & Management**

   * LeadSource
   * LeadStatus
   * LeadAge
   * TotalClientSales
   * ConversionRate

3. **Stock Management & Sales Targeting**

   * ProductID
   * StockLevel
   * RestockAlertThreshold
   * SalesTarget
   * TotalProductSales

4. **Forecasting & Strategy**

   * HistoricalSalesData
   * SeasonalityIndex
   * MarketTrends
   * TotalClientSales
   * SalesGrowthRate

5. **Lead Prioritization**

   * LeadScore
   * TotalClientSales
   * LeadAge
   * ConversionProbability
   * ClientEngagementScore

---

## Machine Learning Models Used

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Machine (SVC)
5. K-Nearest Neighbors (KNN)
6. Decision Tree Classifier

---

## Preprocessing Pipeline

### 1. Train-Test Split

* **Test Size:** 20%
* **Random State:** 42 (for reproducibility)

### 2. Categorical Feature Encoding

* Applied `OneHotEncoder` to categorical columns such as `ClientCode` and `LeadSource`.
* Handled unknown categories using `handle_unknown='ignore'`.

### 3. Standardization

* All numeric features are scaled using `StandardScaler` for normalized input to models.

---

## Results Evaluation

### Accuracy Comparison

For each feature set and model combination, the accuracy on the test dataset is calculated and stored. However, the model training and evaluation section is missing from the script and should be added to populate the `results` dictionary.

### Output Summary

The script prints out a table of model names, feature set names, and corresponding accuracies. It then identifies the best-performing model-feature combination.

---

## Notes and Recommendations

* **Missing Section:** Model training loop is not present. Each model in the `models` dictionary should be trained and evaluated on each feature set to complete the accuracy comparison.

* **Model Persistence:** Consider saving the best model using joblib or pickle.

* **Visualization:** Adding plots such as confusion matrices or ROC curves can enhance model evaluation.

* **Temporal Features:** `LastPurchaseDate` and `HistoricalSalesData` may benefit from datetime processing or feature engineering.

---

## Future Enhancements

1. Integrate additional evaluation metrics (F1-score, precision, recall).
2. Automate feature importance ranking.
3. Deploy the best model with an API for real-time predictions.
4. Include advanced techniques like stacking or automated hyperparameter tuning.

---

## Conclusion

This script sets up a strong foundation for evaluating various sales-related business questions using ML models. Once completed, it will help identify the best strategies for sales growth, client engagement, and operational efficiency.
