# Airline Customer Satisfaction Analysis

## 1. Introduction
Customer satisfaction is one of the performance indicators in the airline industry.  
Increasing competition, price sensitivity, and service differentiation force airlines to understand which factors most strongly influence passenger satisfaction.

Accurately predicting customer satisfaction enables airlines to:

- Identify weaknesses in service quality  
- Improve customer retention and loyalty  
- Optimize operational and service-level decisions  
- Allocate resources more efficiently  

In this project, machine learning techniques are used to predict airline customer satisfaction based on:

- Demographic characteristics  
- Travel information  
- Service quality ratings  
- Delay-related operational variables  

Unlike simple predictive studies, this project places strong emphasis on:

- Proper data preprocessing  
- Avoiding data leakage  
- Fair model comparison  
- Robust validation strategies  
- Model interpretability and explainability  

---

## 2. Dataset Description
The dataset consists of airline customer survey responses combined with operational flight information.  
Each observation corresponds to a single passenger.

### Feature Groups

**1. Demographic Variables**
- Age  
- Gender  
- Customer Type (Loyal / Disloyal)  

**2. Travel Information**
- Type of Travel (Business / Personal)  
- Class (Economy, Economy Plus, Business)  
- Flight Distance  

**3. Service Quality Ratings (1–5 scale)**
- Seat comfort  
- Inflight entertainment  
- On-board service  
- Check-in service  
- Online boarding  
- Ease of online booking  
- Cleanliness  
- Food and drink  
- Leg room service  
- Baggage handling  
- Inflight WiFi service  

**4. Operational Variables**
- Departure delay (minutes)  
- Arrival delay (minutes)  

**Target Variable**
- `satisfaction` (binary: satisfied / dissatisfied)

**Dataset Size**
- Original: 129,880 observations  
- After cleaning: 119,255 observations  

---

## 3. Problem Definition
This project is formulated as a **supervised binary classification problem**.

**Input:**  
Passenger-level features related to demographics, travel behavior, service quality, and operational performance.

**Output:**  
Binary satisfaction label.

**Objective:**  
Learn a mapping \( f(X) \rightarrow y \) that predicts customer satisfaction accurately while maintaining strong generalization performance.

**Evaluation Metrics**
- Accuracy  
- ROC-AUC  
- Precision, Recall, F1-score  

---

## 4. Data Cleaning and Preprocessing

### 4.1 Handling Service Rating Anomalies
Service quality variables are defined on a 1–5 scale.  
Observations with a value of **0** were treated as invalid entries.

**Action Taken**
- Rows containing any `0` in service rating variables were removed.

**Rationale**
- Zero does not represent a meaningful customer evaluation  
- Retaining such values would distort interpretation and model learning  

---

### 4.2 Delay Variable Treatment
Departure and arrival delays contain extreme outliers that can negatively affect model stability.

**Action Taken**
- Winsorization at the **95th percentile**
- Original delay variables replaced with capped versions

**Rationale**
- Preserve relative delay information  
- Reduce influence of extreme, rare events  
- Improve numerical stability without dropping observations  

---

### 4.3 Missing Values
Remaining missing values were inspected.

**Action Taken**
- Observations with missing values were removed.

**Rationale**
- Large dataset size allows safe row-wise deletion  
- Avoids introducing artificial patterns via imputation  

---

### 4.4 Encoding Strategy
Categorical variables were encoded using **One-Hot Encoding**.

**Rationale**
- Avoids artificial ordinal assumptions  
- Compatible with all model families  

---

### 4.5 Scaling Strategy
Standardization (mean = 0, std = 1) was applied **only when required**.

| Model Type            | Scaling Applied |
|----------------------|----------------|
| Logistic Regression  | Yes |
| Neural Networks      | Yes |
| Tree-based Models    | No |

**Rationale**
- Linear and neural models are scale-sensitive  
- Tree-based models are scale-invariant  

---

### 4.6 Data Leakage Prevention
Strict anti-leakage rules were enforced:

- Train / validation / test split performed **before preprocessing**
- All preprocessing steps fitted **only on training data**
- Validation and test sets transformed using fitted objects
- Cross-validation pipelines used unfitted preprocessors

This ensures unbiased and reliable performance estimates.

---

## 5. Exploratory Data Analysis (EDA)

Key findings:

- Loyal customers exhibit significantly higher satisfaction  
- Business travelers are more satisfied than personal travelers  
- Higher travel class strongly correlates with satisfaction  
- Service quality variables dominate satisfaction outcomes  
- Delay variables have a weaker but consistently negative effect  

EDA was used **only for interpretation**, not feature selection, to avoid confirmation bias.

---

## 6. Modeling Approach

### 6.1 Logistic Regression
A strong, interpretable baseline model.

**Configuration**
- L2 regularization  
- Stratified 5-fold cross-validation  
- ROC-AUC as optimization metric  

**Results**
- Accuracy ≈ 0.860  
- ROC-AUC ≈ 0.94  

**Conclusion**
Provides solid baseline performance with excellent interpretability and no overfitting.

---

### 6.2 Decision Tree & Ensemble Methods

#### 6.2.1 Decision Tree
- Interpretable rule-based classifier  
- Captures nonlinearities  

**Results**
- Accuracy ≈ 0.940  
- ROC-AUC ≈ 0.959  

---

#### 6.2.2 Bagging (Bootstrap Aggregating)
- Reduces variance of decision trees  

**Results**
- Accuracy ≈ 0.952  
- ROC-AUC ≈ 0.991  

---

#### 6.2.3 AdaBoost
- Sequentially focuses on misclassified samples  

**Results**
- Accuracy ≈ 0.943  
- ROC-AUC ≈ 0.988  

---

### 6.3 XGBoost
State-of-the-art gradient boosting model.

**Tuning**
- Bayesian optimization using Optuna  

**Results**
- Accuracy ≈ 0.9604  
- ROC-AUC ≈ 0.9944

**Key Features**
- Inflight entertainment  
- Seat comfort  
- Online booking ease  
- Customer loyalty  
- Travel type  

---

### 6.4 Neural Networks
Feedforward neural networks with dropout and early stopping.

**Results**
- Accuracy ≈ 0.948  
- ROC-AUC ≈ 0.994  

**Observation**
- Stable convergence  
- Minimal overfitting  
- Strong nonlinear modeling capability  

---

## 7. Model Comparison

| Model                | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 0.860    | 0.94    |
| Decision Tree       | 0.940    | 0.959   |
| Bagging             | 0.952    | 0.991   |
| AdaBoost            | 0.943    | 0.988   |
| Neural Network      | 0.948    | 0.994   |
| XGBoost             | 0.960    | 0.994  |

All models were evaluated under **identical data splits and leakage-safe pipelines**.

---

## 8. Overfitting and Validation Analysis
- Stratified train/validation/test splits  
- Cross-validation consistency checks  
- Learning curve inspection  

No evidence of data leakage or severe overfitting was observed.

---

## 9. Conclusion
This project demonstrates that airline customer satisfaction can be predicted with high accuracy using modern machine learning techniques.

**Key Takeaways**
- Service quality is the dominant driver of satisfaction  
- Loyal and business customers are more satisfied  
- Ensemble and nonlinear models significantly outperform linear baselines  
- Proper preprocessing and validation are essential  

The project follows academic best practices and provides a fully reproducible, leakage-free machine learning workflow suitable for university-level evaluation and real-world application.
