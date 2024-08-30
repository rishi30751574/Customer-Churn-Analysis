# Customer Churn Prediction Model - README

## Table of Contents
1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Data Dictionary](#data-dictionary)
4. [Tech Stack](#tech-stack)
5. [Steps Followed in Building the ML Model](#steps-followed-in-building-the-ml-model)
6. [Results Explanation](#results-explanation)

---

## 1. Objective <a name="objective"></a>
The goal of this project is to build a machine learning model that predicts customer churn. The dataset consists of customer information, and the objective is to use this data to predict whether a customer is likely to churn (leave) or not. This project also emphasizes the interpretability of the model by analyzing feature importance.

---

## 2. Dataset <a name="dataset"></a>
- **Description**: The dataset includes customer demographics and transactional behavior used to build the churn prediction model.
- **Target**: The target variable is a binary classification of whether a customer churned (Yes/No).
- **Features**: Multiple features like customer tenure, monthly charges, total charges, and other behavioral attributes.
- **Preprocessing**: Missing values were handled, and categorical variables were encoded.

---

## 3. Data Dictionary

- **CLIENTNUM**: Unique identifier for each customer, useful for tracking and referencing individual customers.

- **Attrition_Flag**: Indicates if a customer has churned ("Attrited") or not ("Existing"). This is the target variable for prediction.

- **Customer_Age**: The age of the customer. Age could influence customer behavior and loyalty.

- **Gender**: The gender of the customer, which might impact preferences and behaviors.

- **Dependent_count**: The number of individuals financially dependent on the customer. This can affect a customer's financial stability and decision-making.

- **Education Level**: The highest level of education attained by the customer. Education level might correlate with financial behavior and stability.

- **Marital_Status**: The marital status of the customer, which can influence financial decisions and relationships with the bank.

- **Income_Category**: The income bracket of the customer, providing insight into their financial capacity and stability.

- **Card_Category**: Type of credit card the customer holds. Different cards may come with different benefits and usage patterns.

- **Months_on_books**: The number of months the customer has been with the bank. Longer duration may indicate higher loyalty.

- **Total_Relationship_Count**: Total number of different types of accounts or services that the customer holds with the bank.

- **Months_Inactive_12_mon**: Number of months the customer was inactive in a 12-month period.

- **Contacts_Count_12_mon**: Number of contacts the customer had with the bank in a 12-month period.

- **Credit_Limit**: Maximum amount a lender allows a borrower to spend or borrow on a credit account.

- **Total_Revolving_Bal**: Total amount of outstanding debt on a credit account that carries over from month to month.

- **Avg_Open_To_Buy**: Average amount of available credit a customer has on their credit accounts.

- **Total_Amt_Chng_Q4_Q1**: Change in amount deposited by the customer from Q1 to Q4.

- **Total_Trans_Amt**: Cumulative sum of all individual transactions.

- **Total_Trans_Ct**: Number of transactions by the customer.

- **Total_Ct_Chng_Q4_Q1**: Change in the total count of transactions from Q1 to Q4.

- **Avg_Utilization_Ratio**: Average of their available credit that is used over a given period.

---

## 4. Tech Stack <a name="tech-stack"></a>
The following technologies and libraries were used:

- **Programming Language**: Python
- **Framework**: Scikit-learn
- **Libraries**:
  - **Numpy**: Data manipulation
  - **Pandas**: Data handling
  - **Matplotlib**: Visualization
  - **Logistic Regression**: Classification model from `Scikit-learn`
  - **GridSearchCV**: Hyperparameter tuning

---

## 5. Steps Followed in Building the ML Model <a name="steps-followed-in-building-the-ml-model"></a>

### 1. Data Preprocessing
- **Missing Values**: Handled missing data by removing or imputing where necessary.
- **Encoding**: Categorical features like customer types were target encoded.
- **Normalization**: Numeric features were scaled using `StandardScaler` for optimal model performance.

## 2. Logistic Regression Model

### a. Model Selection and Tuning
- **Base Model**: Logistic Regression was selected as the classification model.
- **Hyperparameter Tuning**: Used GridSearchCV to find the best parameters for `C`, `penalty`, and `solver`.
  - **Cross-validation** was performed using Stratified K-Folds (sklearn) for robust model evaluation.

### b. Model Training
- **Regularization**: L2 regularization was applied to avoid overfitting.
- **Training Data**: The model was trained on the processed dataset with balanced class weights.

---

## 3. XGBoost Model

## a. Model Selection and Tuning
- **Base Model**: XGBoost (`XGBClassifier`) was selected due to its efficiency in handling large datasets and its robustness with imbalanced classes.
- **Hyperparameter Tuning**: The early_boosting_rounds feature in XGBclassifier was used to find the optimal number of trees to avoid overfitting
- **Cross-Validation**: Performed using `StratifiedKFold` to handle class imbalance and ensure robust model evaluation.

## b. Model Training
- **Boosting Method**: Gradient boosting was employed to iteratively improve the model.
- **Training Data**: The model was trained on a processed dataset, with `scale_pos_weight` set to handle class imbalance.

## 4. Model Evaluation
- **Metrics Used**:
  - **Accuracy**: Overall correctness of the model.
  - **F1 Score**: To balance precision and recall, especially for the imbalanced dataset.
- **Cross-validation Score**: Used to assess model generalizability across data splits.
- **Test Performance**: Evaluated on the test set using accuracy and F1 score.

---

## 6. Results Explanation <a name="results-explanation"></a>

### 1. Logistic Regression Performance

#### Cross-Validation Score
- **Regularization Parameter (C)**: 0.5
- **Solver**: `lbfgs`
- **Penalty**: L2
- **Cross-Validation Score**: 0.8

The cross-validation score of 0.8 suggests that the model generalizes well across different folds of the training data.

#### Test Set Performance
- **F1 Score on Test Set**: 0.85

The F1 score of 0.85 on the test set indicates strong performance on unseen data, maintaining a good balance between precision and recall.

---

### 2. XGBoost Performance

#### Cross-Validation Score
- **n_estimators**: 124
- **scale_pos_weight**: 10
- **eval_metric**: AUC
- **Cross-Validation Score**: 0.91

The cross-validation score of 0.91 suggests that the model generalizes well across different folds of the training data.

#### Test Set Performance
- **F1 Score on Test Set**: 0.9

The F1 score of 0.9 on the test set indicates strong performance on unseen data, maintaining a good balance between precision and recall.

### Feature Importance
To interpret the model, feature importance analysis was performed:
- The most important features were identified based on the coefficients of the logistic regression model.
- This analysis helps in understanding which features contribute the most to customer churn prediction, providing actionable business insights.
