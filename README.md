# Heart Disease Prediction Model

## 📌 Overview
This project implements a **Machine Learning pipeline** for predicting the presence of **heart disease** using the **UCI Heart Disease dataset**.  
It includes data preprocessing, model training, hyperparameter tuning, and performance evaluation.

## 🚀 Features
- Data preprocessing (handling missing values, encoding categorical features, scaling).
- Model training with **Logistic Regression, Random Forest, and Support Vector Machine (SVM)**.
- Hyperparameter tuning using **GridSearchCV** and **RandomizedSearchCV**.
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix.
- Final trained model saved for deployment.

## 🛠️ Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project:
   ```bash
   python main.py
   ```

## 📊 Dataset
The dataset used is the **UCI Heart Disease dataset** (ID: 45) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

## 📈 Results
- Best performing model: **Random Forest Classifier**
- Achieved accuracy: ~85% (depending on train-test split and tuning).

## 📂 Project Structure
```
heart-disease-prediction/
│── main.py               # Main script for training and evaluation
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
│── .gitignore            # Git ignore rules
│── models/               # Saved trained models
│── data/                 # Dataset (if stored locally)
```

## 👨‍💻 Author
- Developed by **[Your Name]**  
- For educational and research purposes.

