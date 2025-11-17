# Regression-Classification
Heart Failure Classification &amp; Insurance Cost Regression Models
# 🤖 Machine Learning Models: Classification & Regression

**Live Demo:** [https://yasaausman.github.io/Regression-Classification/](https://yasaausman.github.io/Regression-Classification/)

Interactive web application featuring two machine learning models deployed using PyTorch and ONNX for real-time predictions in the browser.

---

## 📊 Project Overview

This project implements and deploys two complete machine learning pipelines:

1. **Heart Failure Classification** - Binary classification to predict heart disease risk
2. **Insurance Cost Regression** - Continuous prediction of health insurance charges

Both models are trained using PyTorch, exported to ONNX format, and deployed via GitHub Pages for browser-based inference.

---

## 🎯 Models

### 1. Heart Failure Classification

**Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Samples:** 918 patients
- **Features:** 11 medical indicators
- **Task:** Binary classification (Heart Disease: Yes/No)

**Model Performance:**
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Wide Neural Network** ✅ | **89.67%** | **0.8961** | **0.8982** | **0.8967** |
| Deep Neural Network | 88.04% | 0.8801 | 0.8806 | 0.8804 |
| MLP | 85.33% | 0.8521 | 0.8552 | 0.8533 |

**Architecture:**
- Input Layer: 11 features
- Hidden Layers: 64 → 32 neurons
- Output Layer: 2 classes (Binary classification)
- Activation: ReLU
- Regularization: Dropout (0.3-0.4)

---

### 2. Insurance Cost Regression

**Dataset:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Samples:** 1,338 individuals
- **Features:** 6 personal attributes (age, sex, BMI, children, smoker, region)
- **Task:** Regression (predict insurance charges in USD)

**Model Performance:**
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Deep Neural Network** ✅ | **0.8743** | **$4,417** | **$2,198** |
| XGBoost | 0.8681 | $4,526 | $2,458 |
| MLP | 0.8468 | $4,877 | $3,251 |

**Architecture:**
- Input Layer: 6 features
- Hidden Layers: 64 → 32 → 16 neurons
- Output Layer: 1 continuous value
- Activation: ReLU
- Normalization: BatchNorm1d
- Regularization: Dropout (0.2-0.3)

---

## 🛠️ Technologies Used

- **Deep Learning Framework:** PyTorch 2.x
- **Model Export:** ONNX (Open Neural Network Exchange)
- **Browser Inference:** ONNX Runtime Web
- **Data Processing:** 
  - NumPy
  - Pandas
  - scikit-learn (StandardScaler, LabelEncoder)
- **Visualization:** Matplotlib, mlxtend
- **Deployment:** GitHub Pages
- **Additional Models:** XGBoost (comparison)

---

## 📁 Repository Structure

```
Regression-Classification/
├── index.html                          # Main web application
├── heart_failure_model.onnx            # Classification model (ONNX)
├── scaler_params.json                  # Classification preprocessing params
├── label_encoders.json                 # Categorical encoding mappings
├── insurance_regression_model.onnx     # Regression model (ONNX)
├── insurance_scaler_params.json        # Regression preprocessing params
├── insurance_label_encoders.json       # Insurance encoding mappings
└── README.md                           # This file
```

---

## 🚀 How to Use

### **Online (Live Demo):**
Visit [https://yasaausman.github.io/Regression-Classification/](https://yasaausman.github.io/Regression-Classification/)

1. Choose a tab: **Heart Disease Classification** or **Insurance Cost Regression**
2. Fill in the required information
3. Click the predict button
4. View instant AI-powered predictions!

### **Local Development:**
```bash
# Clone the repository
git clone https://github.com/yasaausman/Regression-Classification.git

# Navigate to directory
cd Regression-Classification

# Open with a local server (required for ONNX loading)
python -m http.server 8000

# Visit http://localhost:8000 in your browser
```

---

## 🔬 Methodology

### Data Preprocessing

**Classification:**
1. Categorical encoding (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
2. Standardization using StandardScaler (mean=0, std=1)
3. Train-test split (80-20) with stratification

**Regression:**
1. Categorical encoding (sex, smoker, region)
2. Standardization of features and target
3. Train-test split (80-20, random_state=42)

### Model Training

**Hyperparameters:**
- Learning Rate: 0.001-0.003
- Optimizer: Adam
- Batch Size: 32
- Epochs: 500
- Loss Functions:
  - Classification: CrossEntropyLoss
  - Regression: MSE (Mean Squared Error)

**Training Process:**
1. Multiple architectures tested (MLP, Deep NN, Wide NN)
2. Best model selected based on F1-score (classification) and R² (regression)
3. Model exported to ONNX format for web deployment

---

## 📊 Results & Analysis

### Classification Results

**Confusion Matrix (Best Model - Wide NN):**
```
                Predicted
                No    Yes
Actual   No    [69]   [13]
         Yes   [6]    [96]
```

**Key Insights:**
- High sensitivity (93.6%) - Good at detecting heart disease
- Strong specificity (84.1%) - Low false positive rate
- Balanced performance across both classes

---

### Regression Results

**Performance Visualization:**
- Strong correlation between predicted and actual values (R² = 0.8743)
- MAPE: 22.81% (Mean Absolute Percentage Error)
- Model explains 87.43% of variance in insurance costs

**Key Cost Factors:**
- Smoking status (most significant)
- BMI (Body Mass Index)
- Age
- Number of dependents

---

## 🌟 Features

- ✅ **Real-time Predictions** - Instant inference in the browser
- ✅ **No Backend Required** - Fully client-side using ONNX Runtime Web
- ✅ **Responsive Design** - Works on desktop, tablet, and mobile
- ✅ **Interactive UI** - Tabbed interface with smooth animations
- ✅ **Input Validation** - Ensures data quality before prediction
- ✅ **Detailed Results** - Confidence scores and recommendations
- ✅ **Privacy-Focused** - All predictions run locally in browser

---

## 📈 Model Comparison

### Why Deep Neural Networks?

Both deployed models use deep neural networks due to:
1. **Superior Performance** - Outperformed simpler models
2. **ONNX Compatibility** - Seamless export for web deployment
3. **Generalization** - Strong performance on test data
4. **Efficient Inference** - Fast predictions in browser

### XGBoost Note

While XGBoost achieved competitive performance (R²=0.8681 for regression), it could not be exported to ONNX due to compatibility issues with XGBoost 2.0+. The Deep NN model was chosen for deployment as it:
- Performs slightly better (R²=0.8743)
- Exports successfully to ONNX
- Provides seamless browser-based inference

---

## 🎓 Educational Value

This project demonstrates:
- ✅ Complete ML pipeline (data → training → deployment)
- ✅ Multiple model architectures and comparison
- ✅ Proper preprocessing and standardization
- ✅ Model evaluation with appropriate metrics
- ✅ Production deployment considerations
- ✅ Real-world constraints (ONNX compatibility)

---

## 📚 Datasets

### Heart Failure Prediction
- **Source:** [Kaggle - fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Features:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
- **Target:** HeartDisease (0=No, 1=Yes)

### Insurance Cost
- **Source:** [Kaggle - Medical Cost Personal](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Features:** age, sex, bmi, children, smoker, region
- **Target:** charges (continuous, in USD)

---

## 🔧 Technical Implementation

### ONNX Export Process
```python
# PyTorch to ONNX conversion
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

### Browser Inference
```javascript
// Load and run ONNX model in browser
const session = await ort.InferenceSession.create('model.onnx');
const tensor = new ort.Tensor('float32', input_data, [1, n_features]);
const results = await session.run({ input: tensor });
```

---

## 🤝 Contributing

This is an educational project. Feel free to:
- Fork the repository
- Experiment with different architectures
- Try other datasets
- Improve the UI/UX
- Add more features

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Yasa Ausman**

- GitHub: [@yasaausman](https://github.com/yasaausman)
- Project Link: [https://github.com/yasaausman/Regression-Classification](https://github.com/yasaausman/Regression-Classification)

---

## 🙏 Acknowledgments

- Dataset providers on Kaggle
- PyTorch and ONNX teams
- scikit-learn contributors
- GitHub Pages for free hosting

---

## 📞 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**⭐ If you found this project helpful, please consider giving it a star!**

---

*Last Updated: November 2025*
