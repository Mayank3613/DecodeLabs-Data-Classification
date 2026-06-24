"""
test_classification.py  –  Unit tests for the Iris KNN Classifier.
"""

import pytest
import numpy as np
from data_classification import IrisClassifier


@pytest.fixture
def clf():
    classifier = IrisClassifier(n_neighbors=3, test_size=0.3, random_state=42)
    return classifier


@pytest.fixture
def trained_clf(clf):
    clf.load_data()
    clf.split_data()
    clf.scale_features()
    clf.train()
    return clf


def test_load_data(clf):
    """Test that data loads with correct shapes and features."""
    clf.load_data()
    assert clf.X is not None
    assert clf.y is not None
    assert clf.X.shape == (150, 4)
    assert len(clf.y) == 150
    assert len(clf.feature_names) == 4
    assert list(clf.target_names) == ["setosa", "versicolor", "virginica"]


def test_split_data(clf):
    """Test train-test splitting proportions and stratification."""
    clf.load_data()
    clf.split_data()
    
    # 150 * 0.3 = 45 test, 105 train
    assert clf.X_train.shape == (105, 4)
    assert clf.X_test.shape == (45, 4)
    assert len(clf.y_train) == 105
    assert len(clf.y_test) == 45
    
    # Stratification check (equal representation since classes are balanced)
    assert np.bincount(clf.y_train).tolist() == [35, 35, 35]
    assert np.bincount(clf.y_test).tolist() == [15, 15, 15]


def test_scale_features(clf):
    """Test that StandardScaler centers data to mean ~0 and std ~1."""
    clf.load_data()
    clf.split_data()
    clf.scale_features()
    
    # Check mean and variance of scaled training data
    train_mean = np.mean(clf.X_train_scaled, axis=0)
    train_std = np.std(clf.X_train_scaled, axis=0)
    
    # Should be close to 0 and 1
    np.testing.assert_allclose(train_mean, 0, atol=1e-7)
    np.testing.assert_allclose(train_std, 1, atol=1e-7)


def test_train(clf):
    """Test that training fits the KNN classifier."""
    clf.load_data()
    clf.split_data()
    clf.scale_features()
    
    assert not clf.is_fitted
    clf.train()
    assert clf.is_fitted


def test_evaluate(trained_clf):
    """Test model evaluation metrics and shape of confusion matrix."""
    metrics = trained_clf.evaluate()
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    
    # Scores must be between 0 and 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    
    # Confusion matrix shape (3 classes)
    assert metrics["confusion_matrix"].shape == (3, 3)
    # Sum of entries in CM must equal the test set size (45)
    assert np.sum(metrics["confusion_matrix"]) == 45


def test_predict(trained_clf):
    """Test classification predictions on custom measurements."""
    # Custom values close to setosa average
    pred1 = trained_clf.predict([5.1, 3.5, 1.4, 0.2])
    assert pred1 == "setosa"
    
    # Custom values close to virginica average
    pred2 = trained_clf.predict([6.9, 3.1, 5.4, 2.1])
    assert pred2 == "virginica"
    
    # Checking predictions are string types inside the classes
    assert pred1 in trained_clf.target_names
    assert pred2 in trained_clf.target_names


def test_errors_before_stages(clf):
    """Test that calling methods out of pipeline order raises ValueErrors."""
    with pytest.raises(ValueError):
        clf.split_data()  # Called before loading
        
    clf.load_data()
    with pytest.raises(ValueError):
        clf.scale_features()  # Called before splitting
        
    with pytest.raises(ValueError):
        clf.train()  # Called before scaling
        
    with pytest.raises(ValueError):
        clf.evaluate()  # Called before training
        
    with pytest.raises(ValueError):
        clf.predict([5.0, 3.0, 1.5, 0.2])  # Called before training
        
    clf.split_data()
    clf.scale_features()
    clf.train()
    with pytest.raises(ValueError):
        clf.predict([1.0, 2.0, 3.0])  # Invalid feature count (requires 4)
