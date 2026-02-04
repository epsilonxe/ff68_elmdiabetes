"""
ELM Production Model Builder

Creates a self-contained production model artifact that bundles:
1. ELMClassifier class (self-contained, no external dependencies)
2. Trained model weights
3. Preprocessing pipeline (median imputation, scaling, PCA)
4. DiabetesPredictor wrapper class for easy inference

Output:
- production/elm_production_model.pkl - Production-ready model
- production/14_production_model_report.txt - Documentation
"""

import numpy as np
import pandas as pd
import pickle
import os

# =============================================================================
# Configuration
# =============================================================================
# Source artifacts are in ../results/, outputs go to current directory (production/)
SOURCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.dirname(__file__) or '.'


# =============================================================================
# Self-Contained ELM Classifier (copied from elm_classifier.py)
# =============================================================================
class ELMClassifier:
    """
    Extreme Learning Machine for binary classification.

    Self-contained implementation that can be pickled and used independently.

    Parameters
    ----------
    n_hidden : int
        Number of hidden neurons
    activation : str
        Activation function ('sigmoid', 'tanh', 'relu')
    regularization : float
        L2 regularization parameter (ridge regression)
    random_state : int
        Random seed for reproducibility
    """

    def __init__(self, n_hidden=100, activation='sigmoid', regularization=0.001,
                 random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.regularization = regularization
        self.random_state = random_state
        self.input_weights = None
        self.biases = None
        self.output_weights = None
        self.n_features = None

    def _activation_func(self, x):
        """Apply activation function to hidden layer output."""
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def fit(self, X, y):
        """Train the ELM classifier."""
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        n_samples, self.n_features = X.shape

        rng = np.random.RandomState(self.random_state)
        scale = np.sqrt(2.0 / (self.n_features + self.n_hidden))
        self.input_weights = rng.randn(self.n_features, self.n_hidden) * scale
        self.biases = rng.randn(self.n_hidden) * scale

        H = self._compute_hidden_output(X)
        HtH = H.T @ H
        reg_matrix = self.regularization * np.eye(self.n_hidden)
        self.output_weights = np.linalg.solve(HtH + reg_matrix, H.T @ y)

        return self

    def _compute_hidden_output(self, X):
        """Compute hidden layer output matrix H."""
        linear_output = X @ self.input_weights + self.biases
        return self._activation_func(linear_output)

    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        H = self._compute_hidden_output(X)
        raw_output = H @ self.output_weights

        prob_class1 = 1.0 / (1.0 + np.exp(-np.clip(raw_output, -500, 500)))
        prob_class1 = prob_class1.flatten()
        prob_class0 = 1.0 - prob_class1

        return np.column_stack([prob_class0, prob_class1])

    def predict(self, X):
        """Predict class labels using default threshold 0.5."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_hidden': self.n_hidden,
            'activation': self.activation,
            'regularization': self.regularization,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# =============================================================================
# Diabetes Predictor - Production Wrapper
# =============================================================================
class DiabetesPredictor:
    """
    Production-ready diabetes predictor wrapping preprocessing and model.

    Handles the complete prediction pipeline:
    1. Replace zeros with NaN in medical columns
    2. Median imputation for missing values
    3. StandardScaler transformation
    4. PCA dimensionality reduction
    5. ELM prediction with optimized threshold

    Parameters
    ----------
    model : ELMClassifier
        Trained ELM classifier
    scaler : StandardScaler
        Fitted StandardScaler
    pca : PCA
        Fitted PCA transformer
    median_values : dict
        Median values for imputation {column: median}
    zero_invalid_cols : list
        Columns where 0 represents missing data
    threshold : float
        Decision threshold for classification
    feature_order : list
        Expected order of input features
    """

    def __init__(self, model, scaler, pca, median_values, zero_invalid_cols,
                 threshold, feature_order):
        self.model = model
        self.scaler = scaler
        self.pca = pca
        self.median_values = median_values
        self.zero_invalid_cols = zero_invalid_cols
        self.threshold = threshold
        self.feature_order = feature_order

    def preprocess(self, X_raw):
        """
        Preprocess raw input data.

        Parameters
        ----------
        X_raw : array-like or DataFrame of shape (n_samples, 8)
            Raw input with 8 features in order:
            [Pregnancies, Glucose, BloodPressure, SkinThickness,
             Insulin, BMI, DiabetesPedigreeFunction, Age]

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, 7)
            PCA-transformed features ready for prediction
        """
        # Convert to DataFrame if needed
        if isinstance(X_raw, pd.DataFrame):
            df = X_raw[self.feature_order].copy()
        else:
            df = pd.DataFrame(X_raw, columns=self.feature_order)

        # Step 1: Replace zeros with NaN in medical columns
        for col in self.zero_invalid_cols:
            df.loc[df[col] == 0, col] = np.nan

        # Step 2: Median imputation
        for col, median in self.median_values.items():
            df[col] = df[col].fillna(median)

        # Step 3: StandardScaler transform (keep DataFrame to preserve feature names)
        X_scaled = self.scaler.transform(df)

        # Step 4: PCA transform
        X_pca = self.pca.transform(X_scaled)

        return X_pca

    def predict(self, X_raw):
        """
        Predict diabetes diagnosis (0 or 1).

        Parameters
        ----------
        X_raw : array-like or DataFrame of shape (n_samples, 8)
            Raw input with 8 features

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Binary predictions (0 = No Diabetes, 1 = Diabetes)
        """
        X_transformed = self.preprocess(X_raw)
        probabilities = self.model.predict_proba(X_transformed)[:, 1]
        return (probabilities >= self.threshold).astype(int)

    def predict_proba(self, X_raw):
        """
        Predict probability of diabetes.

        Parameters
        ----------
        X_raw : array-like or DataFrame of shape (n_samples, 8)
            Raw input with 8 features

        Returns
        -------
        probabilities : ndarray of shape (n_samples,)
            Probability of diabetes (class 1)
        """
        X_transformed = self.preprocess(X_raw)
        return self.model.predict_proba(X_transformed)[:, 1]


# =============================================================================
# Helper Functions
# =============================================================================
def save_note(filename, content):
    """Save a note file."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Saved: {filename}")


def save_pickle(filename, obj):
    """Save an object to pickle."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Saved: {filename}")


# =============================================================================
# Main: Build Production Model
# =============================================================================
def main():
    print("=" * 60)
    print("ELM PRODUCTION MODEL BUILDER")
    print("=" * 60)

    # Load existing artifacts
    print("\n1. Loading existing artifacts...")

    # Add parent directory to path to find elm_classifier module for unpickling
    import sys
    parent_dir = os.path.dirname(SOURCE_DIR)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Load transformers
    with open(os.path.join(SOURCE_DIR, 'transformers.pkl'), 'rb') as f:
        transformers = pickle.load(f)

    # Load trained model
    with open(os.path.join(SOURCE_DIR, 'elm_model.pkl'), 'rb') as f:
        model_artifact = pickle.load(f)

    print("  Loaded: transformers.pkl")
    print("  Loaded: elm_model.pkl")

    # Extract components
    scaler = transformers['scaler']
    pca = transformers['pca']
    median_values = transformers['median_values']
    zero_invalid_cols = transformers['zero_invalid_cols']
    feature_names = transformers['feature_names']

    trained_model = model_artifact['model']
    optimal_threshold = model_artifact['optimal_threshold']

    print(f"\n2. Model configuration:")
    print(f"   n_hidden: {trained_model.n_hidden}")
    print(f"   activation: {trained_model.activation}")
    print(f"   regularization: {trained_model.regularization}")
    print(f"   optimal_threshold: {optimal_threshold:.2f}")

    # Create self-contained ELM with same weights
    print("\n3. Creating self-contained ELM classifier...")

    production_elm = ELMClassifier(
        n_hidden=trained_model.n_hidden,
        activation=trained_model.activation,
        regularization=trained_model.regularization,
        random_state=trained_model.random_state
    )

    # Copy trained weights
    production_elm.input_weights = trained_model.input_weights.copy()
    production_elm.biases = trained_model.biases.copy()
    production_elm.output_weights = trained_model.output_weights.copy()
    production_elm.n_features = trained_model.n_features

    print(f"   Copied input_weights: {production_elm.input_weights.shape}")
    print(f"   Copied biases: {production_elm.biases.shape}")
    print(f"   Copied output_weights: {production_elm.output_weights.shape}")

    # Create DiabetesPredictor
    print("\n4. Creating DiabetesPredictor wrapper...")

    predictor = DiabetesPredictor(
        model=production_elm,
        scaler=scaler,
        pca=pca,
        median_values=median_values,
        zero_invalid_cols=zero_invalid_cols,
        threshold=optimal_threshold,
        feature_order=feature_names
    )

    # Verify by running a test prediction
    print("\n5. Verifying predictor with test data...")

    test_df = pd.read_csv(os.path.join(SOURCE_DIR, 'diabetes_prepared.csv'))
    sample = test_df.drop('Outcome', axis=1).head(5)
    true_labels = test_df['Outcome'].head(5).values

    predictions = predictor.predict(sample)
    probabilities = predictor.predict_proba(sample)

    print(f"   Sample predictions: {predictions}")
    print(f"   Sample probabilities: {np.round(probabilities, 3)}")
    print(f"   True labels: {true_labels}")

    # Build production artifact
    print("\n6. Building production artifact...")

    production_artifact = {
        'predictor': predictor,
        'feature_order': feature_names,
        'class_labels': {0: 'No Diabetes', 1: 'Diabetes'},
        'threshold': optimal_threshold,
        'model_info': {
            'n_hidden': trained_model.n_hidden,
            'activation': trained_model.activation,
            'regularization': trained_model.regularization,
            'test_accuracy': 0.681,
            'test_auc': 0.754
        },
        'preprocessing_info': {
            'zero_invalid_cols': zero_invalid_cols,
            'median_values': {k: float(v) for k, v in median_values.items()},
            'n_pca_components': pca.n_components_
        }
    }

    save_pickle('elm_production_model.pkl', production_artifact)

    # Generate report
    print("\n7. Generating production model report...")

    report = f"""ELM PRODUCTION MODEL REPORT
============================

1. OVERVIEW
-----------
This file documents the production-ready ELM diabetes prediction model.

Production Model File: elm_production_model.pkl
Usage Example: elm_usage_example.py

2. MODEL ARCHITECTURE
---------------------
Type: Extreme Learning Machine (ELM)
Hidden Neurons: {trained_model.n_hidden}
Activation Function: {trained_model.activation}
Regularization (L2): {trained_model.regularization}
Decision Threshold: {optimal_threshold:.2f}

3. EXPECTED INPUT FORMAT
------------------------
The model expects 8 features in this exact order:
  1. Pregnancies - Number of pregnancies
  2. Glucose - Plasma glucose concentration (0 = missing)
  3. BloodPressure - Diastolic blood pressure mm Hg (0 = missing)
  4. SkinThickness - Triceps skin fold thickness mm (0 = missing)
  5. Insulin - 2-Hour serum insulin mu U/ml (0 = missing)
  6. BMI - Body mass index (0 = missing)
  7. DiabetesPedigreeFunction - Diabetes pedigree function
  8. Age - Age in years

Note: Zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI
are treated as missing values and will be imputed with training medians.

4. PREPROCESSING PIPELINE
-------------------------
Step 1: Replace zeros with NaN in medical columns
        Columns: {zero_invalid_cols}

Step 2: Median imputation using training set medians
        Glucose: {median_values['Glucose']:.1f}
        BloodPressure: {median_values['BloodPressure']:.1f}
        SkinThickness: {median_values['SkinThickness']:.1f}
        Insulin: {median_values['Insulin']:.1f}
        BMI: {median_values['BMI']:.1f}

Step 3: StandardScaler transformation
        (using fitted scaler from training)

Step 4: PCA transformation (8 -> {pca.n_components_} features)
        (using fitted PCA from training)

Step 5: ELM prediction with threshold {optimal_threshold:.2f}

5. OUTPUT FORMAT
----------------
predict(X):       Returns array of 0/1 predictions
                  0 = No Diabetes, 1 = Diabetes

predict_proba(X): Returns array of probabilities
                  Probability of diabetes (class 1)

6. MODEL PERFORMANCE
--------------------
Test Set Metrics (at threshold {optimal_threshold:.2f}):
  Accuracy:  68.10%
  AUC-ROC:   0.754
  Precision: 52.83%
  Recall:    70.00%
  F1-Score:  0.602

7. PRODUCTION ARTIFACT STRUCTURE
--------------------------------
The elm_production_model.pkl file contains:

{{
    'predictor': DiabetesPredictor,  # Main prediction object
    'feature_order': [...],          # Expected input column order
    'class_labels': {{0: 'No Diabetes', 1: 'Diabetes'}},
    'threshold': {optimal_threshold:.2f},
    'model_info': {{
        'n_hidden': {trained_model.n_hidden},
        'activation': '{trained_model.activation}',
        'regularization': {trained_model.regularization},
        'test_accuracy': 0.681,
        'test_auc': 0.754
    }},
    'preprocessing_info': {{
        'zero_invalid_cols': [...],
        'median_values': {{...}},
        'n_pca_components': {pca.n_components_}
    }}
}}

8. USAGE EXAMPLE
----------------
```python
import pickle
import numpy as np

# Load production model
with open('results/elm_production_model.pkl', 'rb') as f:
    artifact = pickle.load(f)

predictor = artifact['predictor']

# Single patient prediction
patient = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
prediction = predictor.predict(patient)
probability = predictor.predict_proba(patient)

print(f"Prediction: {{artifact['class_labels'][prediction[0]]}}")
print(f"Diabetes probability: {{probability[0]:.1%}}")
```

See elm_usage_example.py for complete usage examples.

9. NOTES
--------
- The model is self-contained and requires only numpy, pandas, and pickle
- No external dependencies on sklearn at inference time for the ELM
- However, sklearn is needed for the scaler and PCA transformers
- The predictor handles missing values (zeros) automatically

================================================================================
"""

    save_note('14_production_model_report.txt', report)

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL BUILD COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - elm_production_model.pkl")
    print(f"  - 14_production_model_report.txt")
    print(f"\nUsage: See elm_usage_example.py")


if __name__ == '__main__':
    main()
