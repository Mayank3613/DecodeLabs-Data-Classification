# DecodeLabs Data Classification AI 🌸

A professional, end-to-end supervised machine learning pipeline using K-Nearest Neighbors (KNN) to classify the classic Iris benchmark dataset. This project demonstrates data loading, feature scaling, model training, validation metrics calculation, and interactive terminal interfaces.

---

## 🌟 Key Features

- **Standardized Scaling**: Implements `StandardScaler` to normalize feature magnitudes and prevent data leakage by fitting exclusively on the training split.
- **Multiclass KNN Classifier**: Utilizes `KNeighborsClassifier` with customizable neighbor parameters ($K$) to classify inputs into three species: Setosa, Versicolor, and Virginica.
- **Rich Metric Evaluation**: Reports Overall Accuracy, Macro-averaged Precision, Recall, and F1-Scores, as well as per-class metrics and a detailed Confusion Matrix.
- **Interactive Predictor Loop**: Prompt-based command line interface supporting real-time custom predictions, dataset statistics display, and dynamic re-training.
- **Automated Verification**: Highly structured unit test suite covering pipeline constraints, scaling accuracy, and classification predictions.

---

## 📁 Repository Structure

```
DecodeLabs-Data-Classification/
│
├── main.py                ← Interactive CLI entry point (handles loop, colors, banners, and training CLI commands)
├── data_classification.py ← Core ML pipeline (data loading, train/test splitting, StandardScaler, and KNN fit)
├── test_classification.py ← Pytest unit test suite verifying ML logic pipeline
├── requirements.txt       ← Project dependencies (scikit-learn, pandas, numpy, pytest)
└── .gitignore             ← Ignores python cache files, virtual envs, and temporary test artifacts
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher is recommended.

### 1. Setup Virtual Environment
Run the following commands to isolate dependencies and set up the project locally:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Application
Start the interactive training and prediction CLI:

```bash
python main.py
```

### 3. Run the Test Suite
Ensure all data transformations and classifications are performing correctly:

```bash
pytest test_classification.py -v
```

---

## ⚡ Interactive CLI Commands

During execution, the CLI supports the following inline commands in the `Sepal Length >` prompt:
- **`stats`**: Displays training/testing split details, feature counts, and standard scaling confirmations.
- **`train`**: Allows you to enter a new value of $K$ (neighbors) to rebuild, re-train, and re-evaluate the KNN model instantly.
- **`exit` / `quit`**: Gracefully terminates the application.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
