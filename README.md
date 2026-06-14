# Customer Churn Prediction Pipeline

An end-to-end Machine Learning pipeline utilizing ensemble learning techniques to predict customer churn based on account metrics and usage behaviors.

## 📊 Project Overview
Customer retention is critical for subscription-based services. This project processes real-world customer usage profiles, addresses class imbalances, engineers predictive features, and deploys tuned classifiers to identify high-risk churn profiles before termination occurs.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning & Tuning:** Scikit-Learn (Random Forest, Gradient Boosting, GridSearchCV)
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Workflow Architecture
1. **Data Ingestion:** Loading customer profile features using Pandas.
2. **Preprocessing:** Handling categorical text attributes via Label Encoding and scaling numeric inputs with `StandardScaler`.
3. **Data Splitting:** Using stratified sampling (`stratify=y`) to maintain true churn distributions across training and testing splits.
4. **Model Optimization:** Implementing `GridSearchCV` to maximize the model's F1-Score (essential for handling imbalanced datasets).
5. **Model Comparison:** Evaluating Random Forest vs. Gradient Boosting side-by-side to deploy the strongest classifier.

## 📈 Key Findings & Results
* **Imbalance Management:** Stratification successfully prevented the model from being biased toward non-churning customers.
* **Feature Drivers:** The feature importance analysis successfully isolated the top usage behaviors driving consumer churn.
* **Artifact Export:** Saved final production weights to `best_churn_model.pkl` and data transformations to `customer_scaler.pkl` for downstream application deployment.

## 🚀 How to Run the Project
1. Clone this repository.
2. Open the notebook in Google Colab or any Jupyter environment.
3. Ensure you have the dataset `churn-bigml-20.csv` in your working directory.
4. Execute the cells sequentially to reproduce the metrics and visualizations.

```

---

