# smart-ml-dashboard
Interactive ML dashboard built with Streamlit for model training, evaluation, and visualization.
# 📊 Smart ML Dashboard

An interactive Machine Learning dashboard built using Streamlit that allows users to upload datasets, apply different ML models, and visualize results with minimal effort.

---

## 🌐 Live Demo
Coming soon...

## 🚀 Features

### 📁 Data Handling

* Upload CSV datasets
* Preview dataset instantly
* Dynamic selection of features and target column

---

## 🤖 Supported Models

### 🔹 Regression Models

* Linear Regression
* Multiple Linear Regression
* Polynomial Regression (with automatic degree selection)

### 🔹 Classification Models

* Random Forest (auto n_estimators tuning)
* Decision Tree
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

---

## 📊 Model Evaluation

### Regression Metrics

* R² Score (Train & Test)
* Mean Squared Error (MSE)

### Classification Metrics

* Accuracy (Train & Test)
* Precision
* Recall
* F1 Score
* Classification Report

---

## 📈 Visualizations

### Regression

* Scatter Plot (Actual vs Predicted)
* Line Graph (Prediction Trend)
* Feature Importance (using coefficients)

### Classification

* Confusion Matrix (auto-adjusted for large class counts)
* Prediction Distribution (Bar Chart)
* Prediction Trend (Line Graph)
* Sample Comparison (Actual vs Predicted)

---

## 🧠 Smart Features

* Automatic hyperparameter tuning:

  * Polynomial degree
  * Random Forest estimators
  * SVM C value
  * KNN neighbors

* Handles complex datasets:

  * Avoids unusable confusion matrices for large classes
  * Displays meaningful alternative visualizations

* Optimized performance:

  * Cached data loading
  * Controlled training loops

---

## 🛠️ Installation

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

---

## ▶️ Run the App

```bash
streamlit run pro.py
```

---

## 📂 Project Structure

```
project/
│── pro.py
│── requirements.txt
│── README.md
```

---

## 📌 Requirements

* Python 3.8+
* Streamlit
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## 🎯 Use Cases

* Quick machine learning experimentation
* Learning ML concepts through visualization
* Testing models on custom datasets
* Comparing model performance manually

---

## ⚠️ Limitations

* No cross-validation implemented
* No automatic feature scaling
* Not a full AutoML system

---

## 🔮 Future Improvements

* Add cross-validation for better model evaluation
* Implement automatic model comparison and best model selection
* Add feature scaling options (StandardScaler / MinMaxScaler)
* Enable downloading predictions as CSV
* Improve UI/UX with better layout and styling
* Add support for more advanced models

---

## 👨‍💻 Author

Built as a practical project to understand Machine Learning workflows and create an interactive data-driven tool.

---

## ⭐ Tip

For best results:

* Use clean and well-prepared datasets
* Choose relevant features
* Avoid highly imbalanced target variables

---

## 📌 License

Free to use for learning and educational purposes.
