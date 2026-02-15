# 🏗️ Concrete Strength Prediction with Machine Learning

```markdown
<div align="center">
  <h1>🏗️ Concrete Strength Prediction with Machine Learning</h1>
  <p><i>Advanced regression models for predicting concrete compressive strength with interactive dashboard</i></p>
</div>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-objective">Objective</a> •
  <a href="#-technologies-used">Technologies</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-data-pipeline">Data Pipeline</a> •
  <a href="#-models-implemented">Models</a> •
  <a href="#-results-and-insights">Results & Insights</a> •
  <a href="#-interactive-dashboard">Dashboard</a> •
  <a href="#-how-to-use">How to Use</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-license">License</a>
</p>

---

## 🔍 Overview

This project implements a **complete Machine Learning pipeline** for predicting concrete compressive strength based on mixture composition and curing time. The solution includes exploratory data analysis, multiple regression models comparison, feature importance analysis, and an **interactive Streamlit dashboard** for real-time predictions.

**Key Highlights:**
- 📊 6 regression models tested and compared
- 🎯 Best model: **Gradient Boosting** (R² = 0.7977)
- 🔬 Comprehensive feature importance analysis
- 🎨 Interactive dashboard with real-time simulator
- 📈 Detailed performance metrics and visualizations

---

## 🎯 Objective

The main objective is to develop a **predictive model** that can accurately estimate concrete compressive strength (MPa) based on:

- **Input Features:**
  - Cement (kg/m³)
  - Blast Furnace Slag (kg/m³)
  - Fly Ash (kg/m³)
  - Water (kg/m³)
  - Superplasticizer (kg/m³)
  - Coarse Aggregate (kg/m³)
  - Fine Aggregate (kg/m³)
  - Age (days)

- **Target Variable:**
  - Concrete Compressive Strength (MPa)

This enables engineers and researchers to:
- ✅ Optimize concrete mixture compositions
- ✅ Predict strength without waiting for physical tests
- ✅ Reduce material waste and costs
- ✅ Accelerate the construction planning process

---

## 🛠️ Technologies Used

### **Core Libraries**
- **Python 3.11+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning models and preprocessing

### **Visualization**
- **Matplotlib** - Static plots and visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive charts for dashboard

### **Dashboard**
- **Streamlit** - Interactive web application framework

### **Development Tools**
- **Jupyter Notebook** - Interactive development environment
- **Git** - Version control

---

## 📁 Project Structure

```
Regression_Applied_to_Materials_Engineering/
│   
│── notebooks/
│   ├── concrete_strength_analysis.ipynb    # Complete analysis notebook
│   │
│   ├── concrete_dashboard/                 # Interactive dashboard
│   │   ├── app.py                          # Streamlit application (5 KB)
│   │   ├── dashboard_data.pkl              # Processed data (895 KB)
│   │   └── models.pkl                      # Trained models (12.9 KB)
│   │
│   └── concrete_models/                    # Saved models and reports
│       ├── concrete_project_report.txt     # Project documentation
│       ├── concrete_project_summary.png    # Visual summary (1.2 KB)
│       ├── concrete_scaler.pkl             # Feature scaler (2 KB)
│       └── concrete_strength_model.pkl     # Main model (403 KB)
│   
│── data/
│    └── concrete_data.csv                  # Original dataset
│
├── README.md                               # This file
├── LICENSE.md                              # License information
└── requirements.txt                        # Python dependencies

```

---

## 🔄 Data Pipeline

### **1. Data Exploration** 📊
- **Dataset:** 1030 samples, 9 features
- **Target distribution:** Right-skewed (20-80 MPa range)
- **Missing values:** None detected
- **Outliers:** Identified and analyzed
- **Correlations:** Strong positive correlation with Cement and Age

### **2. Data Preprocessing** 🔧
- **Feature scaling:** StandardScaler for Linear Regression
- **Train-test split:** 80% training, 20% testing (stratified)
- **Feature engineering:** No additional features needed
- **Data validation:** All features within expected ranges

### **3. Model Training** 🤖
Six regression models were trained and evaluated:

| Model | R² Train | R² Test | MAE Test | RMSE Test |
|-------|----------|---------|----------|-----------|
| **Gradient Boosting** | **0.9247** | **0.7977** | **5.34** | **7.56** |
| Random Forest | 0.9773 | 0.7799 | 5.74 | 7.89 |
| Extra Trees | 1.0000 | 0.7595 | 6.24 | 8.24 |
| XGBoost | 0.9999 | 0.7442 | 6.22 | 8.50 |
| Linear Regression | 0.6147 | 0.6117 | 7.82 | 10.47 |
| Ridge Regression | 0.6147 | 0.6117 | 7.82 | 10.47 |

### **4. Model Selection** ✅
**Gradient Boosting** was selected as the best model based on:
- ✅ Best R² score on test set (0.7977)
- ✅ Lowest MAE (5.34 MPa)
- ✅ Good balance between bias and variance
- ✅ Moderate overfitting (13.7%)
- ✅ Robust performance across different concrete types

---

## 🤖 Models Implemented

### **1. Gradient Boosting Regressor** (Best Model)
- **R² Score:** 0.7977
- **MAE:** 5.34 MPa
- **Key Parameters:**
  - n_estimators: 200
  - max_depth: 5
  - learning_rate: 0.1

### **2. Random Forest Regressor**
- **R² Score:** 0.7799
- **MAE:** 5.74 MPa
- **Key Parameters:**
  - n_estimators: 100
  - max_depth: 15

### **3. Linear Regression**
- **R² Score:** 0.6117
- **MAE:** 7.82 MPa
- Baseline model for comparison

### **Other Models:**
- Extra Trees Regressor
- XGBoost Regressor
- Ridge Regression

---

## 📊 Results and Insights

### **Model Performance**

**Key Findings:**
- 🎯 **Gradient Boosting** achieved the best generalization
- 📈 **79.77%** of variance explained in test set
- 📉 Average prediction error: **±5.34 MPa**
- ⚖️ Overfitting controlled at **13.7%**

### **Feature Importance**

**Top 5 Most Important Features:**
1. **Age** (28.5%) - Curing time is the most critical factor
2. **Cement** (24.3%) - Primary binding material
3. **Water** (15.8%) - Affects hydration process
4. **Superplasticizer** (12.1%) - Improves workability
5. **Fly Ash** (8.9%) - Supplementary cementitious material

### **Model Insights**
- ✅ **Age** is the dominant predictor (doubles strength from 7 to 28 days)
- ✅ **Cement content** shows strong positive correlation
- ✅ **Water-cement ratio** is critical for strength
- ✅ **Supplementary materials** (slag, fly ash) provide moderate improvements
- ✅ **Aggregate composition** has minimal direct impact

---

## 🎨 Interactive Dashboard

### **Dashboard Features**

The project includes a **fully interactive Streamlit dashboard** with 7 pages:

#### **1. 🏠 Home**
- Project overview and key metrics
- Dataset summary statistics
- Quick navigation to all sections

#### **2. 📊 Data Exploration**
- Distribution of concrete strength
- Correlation heatmap
- Feature relationships

#### **3. 🔧 Data Treatment**
- Data cleaning process
- Train-test split information
- Feature scaling details

#### **4. 🤖 Models**
- Comparison of all 6 models
- Performance metrics visualization
- Model selection justification

#### **5. 📈 Interpretation**
- Feature importance analysis
- Model behavior insights
- Prediction confidence intervals

#### **6. 🎯 Simulator** (Interactive!)
- **Real-time strength prediction**
- Adjustable sliders for all 8 features
- Instant results with confidence intervals
- Visual feedback on predictions

#### **7. ✅ Project Criteria**
- Compliance with project requirements
- Model evaluation summary
- Technical documentation

### **Running the Dashboard**

```bash
# Navigate to dashboard directory
cd notebooks/concrete_dashboard

# Run Streamlit app
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## 🚀 How to Use

### **Prerequisites**
- Python 3.11 or higher
- pip package manager

### **Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/Laurentius96/Regression_Applied_to_Materials_Engineering.git
cd concrete-strength-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run Jupyter notebooks:**
```bash
jupyter notebook
```

4. **Launch the dashboard:**
```bash
cd notebooks/concrete_dashboard
streamlit run app.py
```

### **Making Predictions**

**Option 1: Using the Dashboard**
1. Open the Simulator page
2. Adjust the sliders for each ingredient
3. View the predicted strength instantly

**Option 2: Using Python**
```python
import pickle
import numpy as np

# Load the model
with open('concrete_dashboard/models.pkl', 'rb') as f:
    models = pickle.load(f)

model = models['Gradient Boosting']

# Example prediction
features = np.array([[350, 100, 30, 180, 8, 950, 750, 28]])
# [Cement, Slag, Fly Ash, Water, Superplasticizer, Coarse Agg, Fine Agg, Age]

prediction = model.predict(features)
print(f"Predicted Strength: {prediction[0]:.2f} MPa")
```

---

## 📈 Model Evaluation Metrics

### **Performance Summary**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.7977 | Model explains 79.77% of variance |
| **MAE** | 5.34 MPa | Average error of ±5.34 MPa |
| **RMSE** | 7.56 MPa | Root mean squared error |
| **MAPE** | 15.2% | Mean absolute percentage error |
| **Overfitting** | 13.7% | Acceptable generalization gap |

### **Residual Analysis**
- ✅ Residuals approximately normally distributed
- ✅ No systematic bias detected
- ✅ Homoscedasticity confirmed
- ✅ No significant outliers in predictions

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### **Ways to Contribute:**

1. **🐛 Report Bugs**
   - Open an issue describing the bug
   - Include steps to reproduce
   - Provide error messages and screenshots

2. **💡 Suggest Features**
   - Propose new models or techniques
   - Suggest dashboard improvements
   - Share ideas for additional analyses

3. **📝 Improve Documentation**
   - Fix typos or clarify instructions
   - Add examples or tutorials
   - Translate documentation

4. **🔧 Submit Code**
   - Fork the repository
   - Create a feature branch
   - Make your changes
   - Submit a pull request

### **Development Guidelines:**
- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation accordingly

### **Areas for Improvement:**
- 🔄 Add cross-validation for model selection
- 📊 Implement additional visualization techniques
- 🧪 Test with different concrete types
- 🌐 Deploy dashboard to cloud platform
- 📱 Create mobile-responsive interface
- 🔍 Add explainability features (SHAP values)
- 🎯 Implement hyperparameter optimization
- 📉 Add model performance monitoring
- 🔐 Implement data validation checks
- 📚 Create comprehensive API documentation

---

## 📚 References and Resources

### **Dataset Source:**
- [UCI Machine Learning Repository - Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

### **Key Papers:**
- Yeh, I-Cheng. "Modeling of strength of high-performance concrete using artificial neural networks." Cement and Concrete research 28.12 (1998): 1797-1808.

### **Libraries Documentation:**
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

## 📝 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.

**You are free to:**
- ✅ Share — copy and redistribute the material

**Under the following terms:**
- 📌 **Attribution** — Give appropriate credit
- 🚫 **NonCommercial** — Not for commercial use
- 🔒 **NoDerivatives** — No modifications allowed

See [LICENSE.md](LICENSE.md) for full details.

---

## 👤 Author

**Your Name**
- GitHub: [@Laurentius96](https://github.com/Laurentius96)
- LinkedIn: [Lorenzo C. Bianchi](https://www.linkedin.com/in/cb-lorenzo/)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **DNC School** for the project framework and guidance
- **UCI Machine Learning Repository** for providing the dataset
- **Open-source community** for the amazing tools and libraries
- **Scikit-learn team** for the excellent machine learning framework
- **Streamlit team** for the intuitive dashboard framework

---

<div align="center">
  <p>⭐ If you found this project helpful, please give it a star!</p>
  <p>Made with ❤️ and Python</p>
</div>
```

---

## 📋 ARQUIVO REQUIREMENTS.TXT

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
streamlit>=1.28.0
xgboost>=2.0.0
```

---