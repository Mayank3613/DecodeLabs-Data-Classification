"""
data_classification.py  –  Iris dataset classification engine using KNN.

Encapsulates data loading, train-test splitting, feature scaling, model training,
and evaluation metrics using scikit-learn.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


class IrisClassifier:
    """
    KNN-based classification pipeline for the Iris Benchmark dataset.
    """
    
    def __init__(self, n_neighbors: int = 5, test_size: float = 0.3, random_state: int = 42):
        """
        Initialize the classifier hyperparameters.
        
        Args:
            n_neighbors (int): Number of neighbors to use for KNN.
            test_size (float): Proportion of dataset to include in the test split.
            random_state (int): Seed used by the random number generator.
        """
        self.n_neighbors = n_neighbors
        self.test_size = test_size
        self.random_state = random_state
        
        # Scikit-learn models/scalers
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        
        # Data storage
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        
        # Split data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Scaled split data storage
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Fit flag
        self.is_fitted = False

    def load_data(self):
        """
        Load the Iris dataset and extract features, targets, and metadata.
        """
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = [name.replace(" (cm)", "") for name in iris.feature_names]
        self.target_names = iris.target_names

    def split_data(self):
        """
        Shuffle and split the data into training and testing sets.
        
        Uses train_test_split with shuffle=True and random_state to remove order bias.
        """
        if self.X is None or self.y is None:
            raise ValueError("Dataset not loaded. Please call load_data() first.")
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=self.y  # Maintain class proportions in splits
        )

    def scale_features(self):
        """
        Scale the raw features using StandardScaler to balance the feature magnitude.
        Fits only on the training set and transforms both training and test sets.
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not split. Please call split_data() first.")
            
        # Fit scaler on training set, then transform both train and test to prevent data leakage
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train(self):
        """
        Train the KNN classifier using the scaled training features.
        """
        if self.X_train_scaled is None or self.y_train is None:
            raise ValueError("Features not scaled. Please call scale_features() first.")
            
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_fitted = True

    def evaluate(self) -> dict:
        """
        Evaluate the model's performance on the test set.
        
        Returns:
            dict: Dictionary of evaluation metrics (accuracy, precision, recall, f1_score, confusion_matrix).
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Please call train() first.")
            
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate global metrics (macro averaged)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="macro")
        recall = recall_score(self.y_test, y_pred, average="macro")
        f1 = f1_score(self.y_test, y_pred, average="macro")
        
        # Calculate per-class metrics
        per_class_precision = precision_score(self.y_test, y_pred, average=None)
        per_class_recall = recall_score(self.y_test, y_pred, average=None)
        per_class_f1 = f1_score(self.y_test, y_pred, average=None)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1_score": per_class_f1,
            "confusion_matrix": cm
        }

    def predict(self, raw_features: list[float]) -> str:
        """
        Predict the Iris species for a single sample of raw features.
        
        Args:
            raw_features (list[float]): A list of 4 floats representing sepal/petal measurements.
            
        Returns:
            str: The name of the predicted Iris species.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained. Please call train() first.")
            
        if len(raw_features) != 4:
            raise ValueError("Input features must contain exactly 4 measurements.")
            
        # Convert to numpy array and reshape to 2D array
        features_arr = np.array(raw_features).reshape(1, -1)
        
        # Scale using the fitted scaler
        features_scaled = self.scaler.transform(features_arr)
        
        # Predict class label
        pred_label = self.model.predict(features_scaled)[0]
        
        # Return species name
        return self.target_names[pred_label]


if __name__ == "__main__":
    # Test script execution
    bot = IrisClassifier()
    bot.load_data()
    bot.split_data()
    bot.scale_features()
    bot.train()
    metrics = bot.evaluate()
    print("Iris KNN Pipeline Evaluation:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
