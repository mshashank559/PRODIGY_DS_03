# Bank Marketing Decision Tree

This project focuses on preparing and modeling the **Bank Marketing Decision Tree** to predict whether a customer will subscribe to a term deposit. It includes data preprocessing, encoding, feature engineering, and modeling using decision trees.

---

## 📁 Dataset

* **Source**: UCI Machine Learning Repository
* **File Used**: `processed_bank_additional_full.csv`

---

## ⚙️ Steps Completed

### 1. Data Cleaning & Preprocessing

* Removed redundant columns (`duration`, etc.)
* Handled categorical variables (e.g., job, education, contact) using **One-Hot Encoding**
* Addressed missing values and ensured column consistency

### 2. Feature Engineering

* Converted categorical variables into binary features (e.g., `job_admin.`, `job_services`, etc.)

### 3. Decision Tree Classifier

* Model: `DecisionTreeClassifier` from Scikit-learn
* Target Variable: `y` (Subscribed or Not)
* Used job, education, contact, and other encoded variables as features
* Evaluated performance using accuracy and classification metrics

### 4. Tools Used

* Python (pandas, scikit-learn)
* Jupyter Notebook / VS Code

---

## ▶️ How to Run

```bash
# Create and activate virtual environment (if needed)
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook
```

---

## 📌 Notes

* The dataset was transformed to a machine-learning-ready format
* One-Hot Encoded columns like `job_services`, `job_admin.` are derived from the original `job` column

---

## 📂 File Structure

```
├── data
│   ├── bank-additional-full.csv              # Raw dataset
│   └── processed_bank_additional_full.csv    # Cleaned dataset
├── notebooks
│   ├── Decision_Tree_Classifier.ipynb        # Modeling notebook
│   └── EDA_and_Modeling.ipynb                # EDA and preprocessing
├── outputs\figures
│   ├── confusion_matrix.png                  # Model evaluation plot
│   └── decision_tree.png                     # Visualized decision tree
├── src
│   ├── main.py
│   ├── modeling.py                           # Model training logic
│   └── preprocessing.py                      # Data cleaning functions
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## ✅ Next Steps

* Hyperparameter tuning of Decision Tree
* Explore ensemble methods (Random Forest, XGBoost)
* Perform cross-validation

---

## 🧠 Inspiration

This project is based on a real-world banking dataset aiming to enhance direct marketing campaign strategies.

---
