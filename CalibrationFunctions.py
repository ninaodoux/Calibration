#==================================================================================
#===================================== IMPORTS ====================================
#==================================================================================



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report, precision_recall_curve,
    recall_score, precision_score, fbeta_score, f1_score, accuracy_score, 
    balanced_accuracy_score, ConfusionMatrixDisplay, brier_score_loss, log_loss
)
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from sklearn.calibration import calibration_curve, CalibratedClassifierCV, CalibrationDisplay
import logging
import math
import itertools
from IPython.display import display, HTML
from scipy.optimize import minimize
from scipy.special import betaln, logsumexp
from venn_abers import VennAbersCalibrator




#==================================================================================
#========================= CALIBRATION METHODS ====================================
#==================================================================================

class ExponentialCalibration:
    """
    Calibrator that fits:
      P(Y=1 | p) = a * p / (a * p + b * (1-p))
    by matching geometric means of p in each class,
    with robust clipping to avoid log(0).
    """

    def __init__(self, eps: float = 1e-15):
        """
        Initializes the calibrator with a small epsilon value to avoid log(0) issues.
        Sets default values for `a` and `b` as 1.0.
        """
        self.eps = eps
        self.a = 1.0  # Calibration parameter for positive class
        self.b = 1.0  # Calibration parameter for negative class

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """
        Fits the calibration model using geometric means of probabilities in each class.

        Args:
            probs (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels (0 or 1).

        Returns:
            self: The fitted calibrator.
        """
        # Clip probabilities to avoid log(0) and extreme values.
        p = np.clip(probs, self.eps, 1 - self.eps)

        # If both classes are present in the data, calculate calibration parameters.
        # Otherwise, keep default a=b=1.0 (no calibration).
        if np.any(y_true == 1) and np.any(y_true == 0):
            self.a = np.exp(np.mean(np.log(p[y_true == 1])))  # Geometric mean for positive class
            self.b = np.exp(np.mean(np.log(p[y_true == 0])))  # Geometric mean for negative class
        else:
            self.a = self.b = 1.0  # Keep default values if only one class exists
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies the fitted calibration model to adjust probabilities.

        Args:
            probs (np.ndarray): Predicted probabilities.

        Returns:
            np.ndarray: Calibrated probabilities.
        """
        # Clip probabilities to avoid extreme values affecting calculations.
        p = np.clip(probs, self.eps, 1 - self.eps)

        num = self.a * p  # Numerator: adjusted probability using parameter a
        denom = num + self.b * (1 - p)  # Denominator: weighted sum of adjusted probabilities

        return num / denom  # Return final calibrated probability

    # Allow Calibrator.predict_proba to call the predict method
    predict_proba = predict
   

#==================================================================================
#==================================================================================
#==================================================================================

class PolynomialNonDecreasingCalibration:
    """
    Polynomial calibration with monotonicity constraint.
    Ensures that the calibrated probabilities follow a non-decreasing trend.
    """

    def __init__(self, degree=3):
        """
        Initializes the calibrator with a specified polynomial degree.
        
        Args:
            degree (int): Degree of the polynomial used for calibration.
        """
        self.degree = degree
        self.coef_ = None  # Placeholder for polynomial coefficients after fitting

    def _polynomial(self, p, coef):
        """
        Evaluates a polynomial function given coefficients.

        Args:
            p (float or np.ndarray): Input probability values.
            coef (list): Polynomial coefficients.

        Returns:
            float or np.ndarray: Polynomial output values.
        """
        return sum(c * p**i for i, c in enumerate(coef))

    def _objective(self, coef, p, y):
        """
        Defines the objective function for optimization, 
        ensuring probability calibration minimizes log loss.

        Args:
            coef (list): Polynomial coefficients being optimized.
            p (np.ndarray): Input probabilities.
            y (np.ndarray): True labels (0 or 1).

        Returns:
            float: Negative log likelihood of the calibrated probabilities.
        """
        p_cal = self._polynomial(p, coef)  # Compute calibrated probability using polynomial
        p_cal = np.clip(p_cal, 1e-15, 1 - 1e-15)  # Clip to avoid log(0) issues
        return -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))  # Log loss calculation

    def _monotonicity_constraint(self, coef):
        """
        Enforces monotonicity constraint by ensuring the polynomial's derivative 
        is non-negative over the range [0,1].

        Args:
            coef (list): Polynomial coefficients.

        Returns:
            float: Minimum derivative value over test points (should be non-negative).
        """
        test_points = np.linspace(0, 1, 100)  # Generate test points within probability range
        derivs = np.array([
            sum(j * coef[j] * p**(j-1) for j in range(1, len(coef)))  # Compute derivative
            for p in test_points
        ])
        return np.min(derivs)  # Return smallest derivative value to ensure monotonicity

    def fit(self, p, y):
        """
        Fits the calibration model by optimizing polynomial coefficients.

        Args:
            p (np.ndarray): Input probabilities.
            y (np.ndarray): True labels (0 or 1).

        Returns:
            self: The fitted calibrator.
        """
        initial = np.zeros(self.degree + 1)  # Initialize polynomial coefficients
        initial[1] = 1.0  # Set linear term to 1.0 for initial scaling

        # Define optimization constraints to enforce monotonicity
        constraints = [{'type': 'ineq', 'fun': self._monotonicity_constraint}]
        bounds = [(None, None)] * len(initial)  # No explicit bounds on coefficients

        # Perform constrained optimization using SLSQP method
        res = minimize(self._objective, initial, args=(p, y),
                       constraints=constraints, bounds=bounds, method='SLSQP')

        self.coef_ = res.x  # Store optimized polynomial coefficients
        return self

    def predict_proba(self, p):
        """
        Applies the fitted polynomial to compute calibrated probabilities.

        Args:
            p (np.ndarray): Input probabilities.

        Returns:
            np.ndarray: Calibrated probabilities.
        
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted.")  # Ensure coefficients exist before predicting

        return np.clip(self._polynomial(p, self.coef_), 0, 1)  # Clip final probabilities


#==================================================================================
#==================================================================================
#==================================================================================

class HistBinningCalibration:
    """
    Fixed-width histogram binning calibration, robust to edge cases.
    Uses histogram bins to calibrate predicted probabilities.
    """

    def __init__(self, n_bins: int = 10, eps: float = 1e-15):
        """
        Initializes the histogram binning calibration model.
        
        Args:
            n_bins (int): Number of bins for discretizing probabilities.
            eps (float): Small epsilon value to prevent extreme probabilities (0 or 1).
        """
        self.n_bins = n_bins
        self.eps = eps
        self.bin_edges = None  # Stores bin edges after fitting
        self.bin_means = None  # Stores mean probability for each bin

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """
        Fits the calibration model by assigning probabilities to histogram bins.

        Args:
            probs (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels (0 or 1).

        Returns:
            self: The fitted calibrator.
        """
        # 1) Clip probabilities to prevent extreme values (0 or 1),
        #    avoiding numerical instability in calculations.
        p = np.clip(probs, self.eps, 1 - self.eps)
        N = len(p)  # Number of samples

        # 2) Limit number of bins based on sample size to avoid bins with no data.
        max_bins = max(2, int(np.sqrt(N)))  # Maximum bins based on sample size
        n_bins = min(self.n_bins, max_bins)  # Enforce user-defined limit

        # 3) Create evenly spaced bin edges between 0 and 1.
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        # 4) Assign probabilities to bins using digitization.
        bin_idx = np.digitize(p, self.bin_edges, right=False) - 1  # Bin index assignment
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # Ensure indices remain within valid range

        # 5) Compute the mean probability of each bin.
        global_rate = np.mean(y_true)  # Fallback rate if bin has no samples
        counts = np.bincount(bin_idx, minlength=n_bins)  # Count number of samples per bin
        sums = np.bincount(bin_idx, weights=y_true, minlength=n_bins)  # Sum true labels per bin

        # Compute bin means safely, avoiding division errors.
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.where(counts > 0, sums / counts, global_rate)

        self.bin_means = means  # Store computed bin means
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Predicts calibrated probabilities by mapping input probabilities to bin means.

        Args:
            probs (np.ndarray): Predicted probabilities.

        Returns:
            np.ndarray: Calibrated probabilities.
        """
        # Clip probabilities to prevent extreme values.
        p = np.clip(probs, self.eps, 1 - self.eps)

        # Assign probabilities to bins.
        idx = np.digitize(p, self.bin_edges, right=False) - 1  # Find bin index for each probability
        idx = np.clip(idx, 0, len(self.bin_means) - 1)  # Ensure indices are within range

        return self.bin_means[idx]  # Return calibrated probabilities based on bin assignments

    predict_proba = predict




    
#==================================================================================
#==================================================================================
#==================================================================================

class BBQCalibration:
    """
    Bayesian Binning into Quantiles calibration.
    Implements model averaging over multiple quantile-based binning schemes.
    """

    def __init__(self, n_bins_options=None, prior='uniform', gamma=1.0):
        """
        Initializes the BBQ calibration model.
        
        Args:
            n_bins_options (list, optional): List of possible bin counts for quantile-based binning.
            prior (str): Prior type, either 'uniform' (no penalty) or 'complexity' (penalizes bins).
            gamma (float): Penalty factor for complexity-based prior.
        """
        if n_bins_options is None:
            n_bins_options = list(range(50, 201, 10))  # Default range of bin counts
        self.n_bins_options = n_bins_options
        self.prior = prior  # Determines how bins are penalized
        self.gamma = gamma  # Penalty strength for complexity-based prior
        self.models = []  # Stores calibration models
        self.weights = None  # Stores model weights for averaging

    def fit(self, probs, y_true):
        """
        Fits the Bayesian binning calibration model.

        Args:
            probs (np.ndarray): Array of predicted probabilities.
            y_true (np.ndarray): Array of true labels (0 or 1).

        Returns:
            self: The fitted calibration model.
        """
        probs = np.asarray(probs)
        y_true = np.asarray(y_true)
        assert probs.shape == y_true.shape  # Ensure arrays have matching shapes

        log_post = []  # Stores posterior log probabilities
        models = []  # Stores binning models

        for K in self.n_bins_options:
            # 1) Compute raw quantiles for bin edges
            raw = np.quantile(probs, np.linspace(0, 1, K+1))

            # 2) Ensure unique edges and fix endpoints at 0 and 1
            edges = np.unique(np.concatenate([[0.0], raw[1:-1], [1.0]]))
            if len(edges) < 2:
                # Skip models that do not provide useful bins
                continue

            # Calculate effective bin count
            K_eff = len(edges) - 1

            # 3) Assign probabilities to bins
            inds = np.clip(np.digitize(probs, edges, right=True) - 1, 0, K_eff - 1)

            # 4) Compute counts and positive labels per bin
            N = np.bincount(inds, minlength=K_eff)  # Total samples in each bin
            P = np.bincount(inds, weights=y_true, minlength=K_eff)  # Sum of positive labels

            # 5) Compute marginal log-likelihood using Beta distribution
            log_marg = np.sum(betaln(P + 1, N - P + 1))

            # 6) Apply prior to penalize bin complexity (if selected)
            if self.prior == 'uniform':
                log_pr = 0.0  # No penalty
            else:  # Complexity-based prior
                log_pr = -self.gamma * K_eff  # Penalizes excessive bins

            # Store posterior log probability and model details
            log_post.append(log_pr + log_marg)
            models.append({'edges': edges, 'N': N, 'P': P})

        # Normalize posterior probabilities
        log_post = np.array(log_post)
        self.weights = np.exp(log_post - logsumexp(log_post))  # Convert to probability weights
        self.models = models  # Store binning models
        return self

    def predict(self, probs):
        """
        Predicts calibrated probabilities using model averaging.

        Args:
            probs (np.ndarray): Array of predicted probabilities.

        Returns:
            np.ndarray: Calibrated probabilities.
        """
        probs = np.asarray(probs)
        p_cal = np.zeros_like(probs, dtype=float)  # Initialize calibrated probability array

        # Compute weighted average of bin-calibrated probabilities
        for w, m in zip(self.weights, self.models):
            inds = np.clip(np.digitize(probs, m['edges'], right=True) - 1, 0, len(m['N']) - 1)
            p_bin = (m['P'][inds] + 1) / (m['N'][inds] + 2)  # Smoothed bin probability estimate
            p_cal += w * p_bin  # Weighted contribution of each model

        return p_cal  





#==================================================================================
#========================== PREPROCESSING CLASS====================================
#==================================================================================
# PYTHON CLASS FOR LOADING SELECTION OF SAMPLE SIZE AND RATIO OF IMBALANCE




class DataPreprocessor:
    """
    A class for preprocessing data, including standardization 
    and splitting into training, test, and calibration sets.
    """
    
    def __init__(self, target_col='INSERT NAME', random_state=42):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        target_col : str, default='INSERT NAME'
            The name of the target column
        random_state : int, default=234
            Random seed for reproducibility
        """
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.original_data = None  # Store original data before any sampling
        
    def load_data(self, file_path):
        """
        Load data from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        self : returns an instance of self
        """
        self.data = pd.read_csv(file_path)
        self.original_data = self.data.copy()  # Store original data
        print(f"Data loaded from {file_path}: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Show target variable distribution
        if self.target_col in self.data.columns:
            target_counts = self.data[self.target_col].value_counts()
            print(f"\nTarget variable distribution:")
            for value, count in target_counts.items():
                print(f"  Class {value}: {count} samples ({count/len(self.data)*100:.2f}%)")
        
        return self

    def sample_dataset(self, sample_size=None, sample_fraction=None):
        """
        Sample the dataset to a specific size or fraction of the original.
        
        Parameters:
        -----------
        sample_size : int, optional
            The absolute size of the sample. If provided, takes precedence over sample_fraction.
        sample_fraction : float, optional
            The fraction of the original dataset to sample (between 0 and 1).
            
        Returns:
        --------
        self : returns an instance of self
        """
        if sample_size is None and sample_fraction is None:
            print("No sampling performed as both sample_size and sample_fraction are None.")
            return self
            
        original_size = len(self.data)
        
        if sample_size is not None:
            # Sample to absolute size
            if sample_size >= original_size:
                print(f"Requested sample size ({sample_size}) is larger than or equal to original size ({original_size}). No sampling performed.")
                return self
                
            self.data = self.data.sample(sample_size, random_state=self.random_state)
            print(f"Sampled dataset to {sample_size} samples ({sample_size/original_size*100:.2f}% of original)")
            
        elif sample_fraction is not None:
            # Sample to fraction of original
            if not 0 < sample_fraction < 1:
                print(f"Sample fraction must be between 0 and 1. Got {sample_fraction}. No sampling performed.")
                return self
                
            sample_size = int(original_size * sample_fraction)
            self.data = self.data.sample(sample_size, random_state=self.random_state)
            print(f"Sampled dataset to {sample_size} samples ({sample_fraction*100:.2f}% of original)")
        
        # Show target variable distribution after sampling
        if self.target_col in self.data.columns:
            target_counts = self.data[self.target_col].value_counts()
            print(f"\nTarget variable distribution after sampling:")
            for value, count in target_counts.items():
                print(f"  Class {value}: {count} samples ({count/len(self.data)*100:.2f}%)")
                
        return self
    
    def prepare_data(self, test_size=0.2, calib_size=0.25):
        """
        Prepare data by splitting into training, test, and calibration sets.
        
        Parameters:
        -----------
        test_size : float, default=0.2
            Proportion of data to be used for testing
        calib_size : float, default=0.25
            Proportion of training data to be used for calibration
            
        Returns:
        --------
        self : returns an instance of self
        """
        print(f"Data shape before splitting: {self.data.shape}")
        print(f"Features (X) shape before splitting: {self.data.drop(self.target_col, axis=1).shape}")

        # Split features and target
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate calibration set from training set
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_temp, y_temp, test_size=calib_size, random_state=self.random_state, stratify=y_temp
        )
        
        # Store the splits
        self.X_train = X_train
        self.X_test = X_test
        self.X_calib = X_calib
        self.y_train = y_train
        self.y_test = y_test
        self.y_calib = y_calib
        
        # Create dataframes for each split
        self.train_df = X_train.copy()
        self.train_df[self.target_col] = y_train
        
        self.test_df = X_test.copy()
        self.test_df[self.target_col] = y_test
        
        self.calib_df = X_calib.copy()
        self.calib_df[self.target_col] = y_calib
        
        # Print dataset sizes
        print(f"\nDataset sizes:")
        print(f"Full dataset: {len(self.data)} samples")
        print(f"Training set: {len(self.train_df)} samples ({len(self.train_df)/len(self.data)*100:.1f}%)")
        print(f"Test set: {len(self.test_df)} samples ({len(self.test_df)/len(self.data)*100:.1f}%)")
        print(f"Calibration set: {len(self.calib_df)} samples ({len(self.calib_df)/len(self.data)*100:.1f}%)")
        
        return self
    
    def balance_target_variable(self, method='undersample', target_ratio=None, majority_class=0):
        """
        Balance the target variable in the TRAINING SET ONLY to control class imbalance.
        Note: This method should be called AFTER prepare_data().
        
        Parameters:
        -----------
        method : str, default='undersample'
            Method to use for balancing: 'undersample' or 'ratio'
        target_ratio : float, optional
            When method='ratio', specifies the desired proportion of minority class samples
            compared to majority class. For example, 0.2 means the dataset will have
            minority:majority ratio of 1:5 (or minority will be 20% of majority class).
        majority_class : int or str, default=0
            The value that represents the majority class in the target variable
            
        Returns: 
        --------
        self : returns an instance of self
        """
        if not hasattr(self, 'train_df'):
            print("Error: Must call prepare_data() before balance_target_variable() to split the data first.")
            return self
            
        # Work only with the training set
        train_data = self.train_df.copy()
        
        if self.target_col not in train_data.columns:
            print(f"Target column '{self.target_col}' not found in the dataset. No balancing performed.")
            return self
            
        # Get current class distribution
        target_counts = train_data[self.target_col].value_counts()
        
        if len(target_counts) < 2:
            print("Training dataset has only one class. Cannot balance.")
            return self
            
        # Find minority and majority classes
        if majority_class in target_counts.index:
            minority_class = [cls for cls in target_counts.index if cls != majority_class][0]
            majority_count = target_counts[majority_class]
            minority_count = target_counts[minority_class]
        else:
            # If specified majority class doesn't exist, use the actual majority
            majority_class = target_counts.index[0]
            minority_class = target_counts.index[1]
            majority_count = target_counts[majority_class]
            minority_count = target_counts[minority_class]
            
        print(f"\nOriginal class distribution in training set:")
        print(f"  Majority class ({majority_class}): {majority_count} samples ({majority_count/len(train_data)*100:.2f}%)")
        print(f"  Minority class ({minority_class}): {minority_count} samples ({minority_count/len(train_data)*100:.2f}%)")
        print(f"  Current ratio (minority:majority): 1:{majority_count/minority_count:.1f}")
        
        # Apply balancing method
        if method == 'undersample':
            # Undersample majority class to match minority class
            majority_samples = train_data[train_data[self.target_col] == majority_class].sample(
                minority_count, random_state=self.random_state
            )
            minority_samples = train_data[train_data[self.target_col] == minority_class]
            balanced_train_data = pd.concat([majority_samples, minority_samples])
            
        elif method == 'ratio':
            if target_ratio is None or target_ratio <= 0:
                raise ValueError("`target_ratio` must be a positive number.")
            total = len(train_data)
            # target_ratio = minority_count / majority_count
            # so minority should be: total * (target_ratio / (1 + target_ratio))
            desired_minority = int(total * target_ratio / (1 + target_ratio))
            desired_majority = total - desired_minority

            # original counts
            orig_counts = train_data[self.target_col].value_counts()
            maj_cls = majority_class if majority_class in orig_counts.index else orig_counts.idxmax()
            min_cls = [c for c in orig_counts.index if c != maj_cls][0]
            orig_maj = orig_counts[maj_cls]
            orig_min = orig_counts[min_cls]

            # sample the minority
            if desired_minority <= orig_min:
                minority_samples = train_data[train_data[self.target_col]==min_cls] \
                                    .sample(desired_minority, random_state=self.random_state)
            else:
                minority_samples = train_data[train_data[self.target_col]==min_cls] \
                                    .sample(desired_minority, replace=True, random_state=self.random_state)

            # sample the majority
            if desired_majority <= orig_maj:
                majority_samples = train_data[train_data[self.target_col]==maj_cls] \
                                    .sample(desired_majority, random_state=self.random_state)
            else:
                majority_samples = train_data[train_data[self.target_col]==maj_cls] \
                                    .sample(desired_majority, replace=True, random_state=self.random_state)

            # combine and shuffle
            balanced_train_data = pd.concat([majority_samples, minority_samples]) \
                                .sample(frac=1, random_state=self.random_state) \
                                .reset_index(drop=True)

            self.train_df = balanced_train_data
            self.X_train = self.train_df.drop(self.target_col, axis=1)
            self.y_train = self.train_df[self.target_col]

            print(f"\nBalanced to ratio {target_ratio:.3f}: "
                f"{maj_cls}={desired_majority}, {min_cls}={desired_minority} "
                f"(total {total})")

    
    def save_datasets(self, output_dir=None, sample_size=None, ratio=None):
        """
        Save the processed datasets to CSV files.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the CSV files. If None, uses a default relative path.
        sample_size : int, optional
            Sample size used, to include in filenames
        ratio : float or str, optional
            Ratio used, to include in filenames
            
        Returns:
        --------
        self : returns an instance of self
        """
        # Set default output directory if not provided
        if output_dir is None:
            # Use a relative path from current working directory
            output_dir = os.path.join("data", "scenarios")
        
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create base filename using sample_size and ratio if provided
        if sample_size is not None and ratio is not None:
            base_filename = f"sample_{sample_size}_ratio_{ratio}"
        elif sample_size is not None:
            base_filename = f"sample_{sample_size}"
        elif ratio is not None:
            base_filename = f"ratio_{ratio}"
        else:
            base_filename = "processed"
        
        # Save train, test, and calibration datasets
        train_path = os.path.join(output_dir, f"{base_filename}_train.csv")
        test_path = os.path.join(output_dir, f"{base_filename}_test.csv")
        calib_path = os.path.join(output_dir, f"{base_filename}_calibration.csv")
        
        self.train_df.to_csv(train_path, index=False)
        self.test_df.to_csv(test_path, index=False)
        self.calib_df.to_csv(calib_path, index=False)
        
        print(f"Train dataset saved to {train_path}")
        print(f"Test dataset saved to {test_path}")
        print(f"Calibration dataset saved to {calib_path}")
        
        return self





#==================================================================================
#======================== FUNCTIONS FOR MODEL EVAL ================================
#==================================================================================

# 1 - Get confusion Matrix and eval
# ===========================================================

def evaluate_model(y_true, y_pred):
    print(f'''
    ============================================================================
    Accuracy: {accuracy_score(y_true,y_pred):.5f}
    Balanced Accuracy: {balanced_accuracy_score(y_true,y_pred):.5f}
    F2 score: {fbeta_score(y_true,y_pred, beta=2):.5f}
    F1 score: {f1_score(y_true,y_pred):.5f}
    Precision: {precision_score(y_true,y_pred):.5f}
    Recall: {recall_score(y_true,y_pred):.5f}
    ============================================================================
    ''')
    
    plot_cmatrix(y_true, y_pred)

def plot_cmatrix(y_true, y_pred, title='Confusion Matrix', figsize=(20,6)):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    ConfusionMatrixDisplay(confusion_matrix).from_predictions(y_true,y_pred, cmap='Blues', values_format=',.0f', ax=ax1)
    ConfusionMatrixDisplay(confusion_matrix).from_predictions(y_true,y_pred, cmap='Blues', normalize='true', values_format='.2%', ax=ax2)
    ax1.set_title(f'{title}', fontdict={'fontsize':18})
    ax2.set_title(f'{title} - Normalized', fontdict={'fontsize':18})
    ax1.set_xlabel('Predicted Label',fontdict={'fontsize':15})
    ax2.set_xlabel('Predicted Label',fontdict={'fontsize':15})
    ax1.set_ylabel('True Label',fontdict={'fontsize':15})
    ax2.set_ylabel('True Label',fontdict={'fontsize':15})

    plt.show()



# 3 - Get AUC and AUC graph 
# ===========================================================
def plot_roc_gini(y_true=None, y_pred=None, size_figure=(6, 6), title='Curva ROC', model_name='Model'):
    ''' 
    ----------------------------------------------------------------------------------------------------------
    Función plot_roc_gini:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función grafica la curva ROC en base a datos reales y predicciones de la variable objetivo.
        
    - Inputs:
        - y_true (list): Lista con los valores reales de la variable objetivo.
        - y_pred (list): Lista con las probabilidades predecidas por el modelo.
        - size_figure (tuple): Tamaño deseado para los gráficos.
        - title (str): Título del gráfico.
        - model_name (str): Nombre del modelo para mostrar en la leyenda.
        
    - Return:
        - dict: Diccionario con AUC, Gini, mejor umbral y G-mean.
    ''' 

    
    if y_true is None or y_pred is None:
        print('\nFaltan parámetros necesarios para la función.')
        return None

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Calculate Gini coefficient
    gini = (2 * roc_auc) - 1
    
    # Calculate geometric means
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # Find the optimal threshold
    ix = np.argmax(gmeans)

    # Create the plot
    plt.figure(figsize=size_figure)
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', 
                label=f'Best Threshold ({thresholds[ix]:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print('\n======================================================================')
    print(f'El coeficiente de GINI es: {gini:.4f}')
    print(f'El área bajo la curva ROC (AUC) es: {roc_auc:.4f}')
    print(f'El mejor Threshold es {thresholds[ix]:.4f}, con G-Mean {gmeans[ix]:.4f}')
    print('======================================================================')
    
    # Return useful metrics
    return {
        'auc': roc_auc,
        'gini': gini,
        'best_threshold': thresholds[ix],
        'best_gmean': gmeans[ix],
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


# 4 - Plot Recall and precision to see optimal threshold 
# ===========================================================
def plot_recall_precision(y_true, y_pred_proba, figsize=(15, 5), threshold_range=(0.01, 0.99, 0.01)):
    """
    Plot recall, precision, F1, F2, and F0.5 scores across different thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_pred_proba : array-like
        Probability estimates of the positive class.
    figsize : tuple, optional (default=(15, 5))
        Figure size (width, height) in inches.
    threshold_range : tuple, optional (default=(0.01, 0.99, 0.01))
        Range for threshold values as (start, stop, step).
    
    Returns:
    --------
    matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # Calculate scores for different thresholds
    recall_precision = []
    for threshold in np.arange(*threshold_range):
        given_threshold = [1 if value > threshold else 0 for value in y_pred_proba]
        recall_precision.append([
            threshold, 
            recall_score(y_true, given_threshold),
            precision_score(y_true, given_threshold),
            fbeta_score(y_true, given_threshold, beta=2),
            fbeta_score(y_true, given_threshold, beta=0.5),
            f1_score(y_true, given_threshold)
        ])
    
    # Create plot
    plt.figure(figsize=figsize)
    ax = sns.pointplot(x=[round(element[0], 2) for element in recall_precision], 
                     y=[element[1] for element in recall_precision],
                     color="red", label='Recall', scale=1)
    ax = sns.pointplot(x=[round(element[0], 2) for element in recall_precision], 
                     y=[element[2] for element in recall_precision],
                     color="blue", label='Precision')
    ax = sns.pointplot(x=[round(element[0], 2) for element in recall_precision], 
                     y=[element[3] for element in recall_precision],
                     color="gold", label='F2 Score', lw=2)
    ax = sns.pointplot(x=[round(element[0], 2) for element in recall_precision], 
                     y=[element[4] for element in recall_precision],
                     color="pink", label='F0.5 Score', lw=1)
    ax = sns.pointplot(x=[round(element[0], 2) for element in recall_precision], 
                     y=[element[5] for element in recall_precision],
                     color="limegreen", label='F1 Score', lw=1)
    
    # Find optimal F2 score
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f2_score = ((1+(2**2)) * precision * recall) / ((2**2) * precision + recall)
    ix = np.argmax(f2_score)
    
    # Highlight best F2 score
    ax.scatter((round(thresholds[ix], 2)*100) if ix < len(thresholds) else 0, 
               f2_score[ix], s=100, marker='o', color='black', 
               label=f'Best F2 (th={thresholds[ix] if ix < len(thresholds) else 0:.3f}, f2={f2_score[ix]:.3f})', 
               zorder=2)
    
    # Set labels
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Probability')
    ax.legend()
    
    # Clean up x-tick labels
    labels = ax.get_xticklabels()
    for i, l in enumerate(labels):
        if (i%5 != 4):  # Keep only every 5th label
            labels[i] = ''
    ax.set_xticklabels(labels, rotation=45, fontdict={'size': 10})
    
    plt.tight_layout()
    plt.show()
    
    return ax



# 5 - Plot Scatter and Histogram of ranking with both thresholds 
# ==============================================================


def stratified_classification_plot(model, X_test, y_test, sample_size=1500):
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Get binary predictions (0/1)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Create a dataframe
    results_df = pd.DataFrame({
        'true': y_test,
        'pred': y_pred,
        'probability': y_pred_proba * 100  # Convert to percentage
    })

    # Add classification result (TP, TN, FP, FN)
    def get_result(row):
        if row['true'] == 1 and row['pred'] == 1:
            return 'TP'
        elif row['true'] == 0 and row['pred'] == 0:
            return 'TN'
        elif row['true'] == 0 and row['pred'] == 1:
            return 'FP'
        else:
            return 'FN'

    results_df['result'] = results_df.apply(get_result, axis=1)

    # Sample stratification
    result_counts = results_df['result'].value_counts()
    sample_counts = {}

    for result, count in result_counts.items():
        proportion = count / len(results_df)
        sample_counts[result] = max(1, int(proportion * sample_size))

    total = sum(sample_counts.values())
    if total < sample_size:
        majority_category = result_counts.index[0]
        sample_counts[majority_category] += (sample_size - total)
    elif total > sample_size:
        majority_category = result_counts.index[0]
        sample_counts[majority_category] -= (total - sample_size)

    sampled_dfs = []
    for result, sample_size in sample_counts.items():
        group_df = results_df[results_df['result'] == result]
        if len(group_df) <= sample_size:
            sampled_dfs.append(group_df)
        else:
            sampled_dfs.append(group_df.sample(n=sample_size, random_state=42))

    stratified_sample = pd.concat(sampled_dfs).reset_index(drop=True)

    # Scatter Plot
    markers = {'TP': '^', 'TN': 'o', 'FP': 's', 'FN': 'X'}
    colors = {'TP': 'darkgreen', 'TN': 'lightgreen', 'FP': 'darkred', 'FN': '#FF3333'}

    plt.figure(figsize=(12, 2))
    sns.set_style("whitegrid")

    for result, group in stratified_sample.groupby('result'):
        y_values = np.random.uniform(0.1, 0.9, size=len(group))
        plt.scatter(
            group['probability'], 
            y_values,  
            label=result,
            marker=markers[result],
            color=colors[result],
            alpha=0.7,
            s=80
        )

    legend_handles = [
        Line2D([0], [0], marker=markers[result], color='w', markerfacecolor=colors[result], 
               markersize=8, label=f"{result} (n={len(stratified_sample[stratified_sample['result'] == result])})")
        for result in ['TP', 'TN', 'FP', 'FN']
    ]

    plt.axvline(x=50, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Decision Threshold (50%)')
    plt.xlabel('Predicted Probability (%)', fontsize=12)
    plt.xlim(0, 100)
    plt.yticks([])
    plt.xticks(range(0, 101, 10))
    plt.legend(handles=legend_handles, title='Result Type', fontsize=10, loc='upper right')
    plt.savefig('stratified_classification_results_spread.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 100, 21)

    for result, color in colors.items():
        subset = stratified_sample[stratified_sample['result'] == result]
        if len(subset) > 0:
            plt.hist(
                subset['probability'],
                bins=bins,
                label=f"{result} (n={len(subset)})",
                color=color,
                edgecolor='black',
                alpha=0.7
            )

    plt.axvline(x=50, color='black', linestyle='--', linewidth=2.5, alpha=0.7, label='Decision Threshold (50%)')
    plt.yscale('log')
    plt.xlabel('Predicted Probability (%)', fontsize=12)
    plt.ylabel('Number of Samples (Log Scale)', fontsize=12)
    plt.title('Distribution of Predicted Probabilities by Result Type', fontsize=14)
    plt.xlim(0, 100)
    plt.legend(title='Result Type', fontsize=10, loc='upper right')
    plt.savefig('stratified_classification_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Confusion Matrix & Classification Report
    conf_matrix = confusion_matrix(stratified_sample['true'], stratified_sample['pred'])
    print("\nConfusion Matrix (Stratified Sample):")
    print(conf_matrix)
    print("\nClassification Report (Stratified Sample):")
    print(classification_report(stratified_sample['true'], stratified_sample['pred']))









#==================================================================================
#============================== ModelEvaluator ====================================
#==================================================================================

# Class with the previously defined functions for evaluation 



class ModelEvaluator:
    """
    A class for comprehensive model evaluation in classification tasks.
    
    This class provides methods to:
    - Calculate and display confusion matrix and common metrics
    - Plot ROC curve and calculate AUC/Gini coefficients
    - Visualize precision-recall trade-offs at different thresholds
    - Display classification results with stratified visualization
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator class."""

class ModelEvaluator:
    def evaluate_metrics(self, y_true, y_pred):
        """
        Calculate and display common classification metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels.
        y_pred : array-like
            Predicted binary labels.
        """
        print(f'''
        =======================================================================
        Accuracy: {accuracy_score(y_true, y_pred):.5f}
        Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.5f}
        F2 Score: {fbeta_score(y_true, y_pred, beta=2):.5f}
        F1 Score: {f1_score(y_true, y_pred):.5f}
        Precision: {precision_score(y_true, y_pred):.5f}
        Recall: {recall_score(y_true, y_pred):.5f}
        =======================================================================
        ''')
        
        self.plot_confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix', figsize=(20, 6)):
        """
        Plot confusion matrix in both raw and normalized formats.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels.
        y_pred : array-like
            Predicted binary labels.
        title : str, optional (default='Confusion Matrix')
            Title for the plots.
        figsize : tuple, optional (default=(20, 6))
            Figure size (width, height) in inches.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Raw Confusion Matrix
        disp1 = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, cmap='Blues', values_format=',.0f', ax=axes[0]
        )
        disp1.plot(ax=axes[0], cmap='Blues', values_format=',.0f', colorbar=False)
        axes[0].set_title(f'{title}', fontsize=18)
        axes[0].set_xlabel('Predicted Label', fontsize=15)
        axes[0].set_ylabel('True Label', fontsize=15)
        axes[0].grid(False)  # Remove grid lines

        # Normalized Confusion Matrix
        disp2 = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, cmap='Blues', normalize='true', values_format='.2%', ax=axes[1]
        )
        disp2.plot(ax=axes[1], cmap='Blues', values_format='.2%', colorbar=False)
        axes[1].set_title(f'{title} - Normalized', fontsize=18)
        axes[1].set_xlabel('Predicted Label', fontsize=15)
        axes[1].set_ylabel('True Label', fontsize=15)
        axes[1].grid(False)  # Remove grid lines

        plt.tight_layout()
        plt.show()

    
    def plot_roc_curve(self, y_true, y_pred_proba, figsize=(8, 8), title='ROC Curve', model_name='Model'):
        """
        Plot ROC curve and calculate AUC/Gini metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels.
        y_pred_proba : array-like
            Probability estimates of the positive class.
        figsize : tuple, optional (default=(8, 8))
            Figure size (width, height) in inches.
        title : str, optional (default='ROC Curve')
            Title for the plot.
        model_name : str, optional (default='Model')
            Name of the model to show in the legend.
            
        Returns:
        --------
        dict
            Dictionary with AUC, Gini, best threshold and G-mean metrics.
        """
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Gini coefficient
        gini = (2 * roc_auc) - 1
        
        # Calculate geometric means
        gmeans = np.sqrt(tpr * (1 - fpr))
        
        # Find the optimal threshold
        ix = np.argmax(gmeans)
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black',
                   label=f'Best Threshold ({thresholds[ix]:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print('\n======================================================================')
        print(f'GINI coefficient: {gini:.4f}')
        print(f'Area Under the ROC Curve (AUC): {roc_auc:.4f}')
        print(f'Best Threshold: {thresholds[ix]:.4f}, with G-Mean: {gmeans[ix]:.4f}')
        print('======================================================================')
        
        # Return useful metrics
        return {
            'auc': roc_auc,
            'gini': gini,
            'best_threshold': thresholds[ix],
            'best_gmean': gmeans[ix],
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    
    def stratified_classification_plot(self, model, X_test, y_test, threshold=0.5, sample_size=1500):
        """
        Generate stratified visualization of classification results.
        
        Parameters:
        -----------
        model : estimator object
            Fitted model with predict_proba method.
        X_test : array-like
            Test features.
        y_test : array-like
            True binary labels.
        threshold : float, optional (default=0.5)
            Classification threshold for binary prediction.
        sample_size : int, optional (default=1500)
            Number of points to include in the visualization.
        """
        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Get binary predictions (0/1)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Create a dataframe
        results_df = pd.DataFrame({
            'true': y_test,
            'pred': y_pred,
            'probability': y_pred_proba * 100  # Convert to percentage
        })
        
        # Add classification result (TP, TN, FP, FN)
        def get_result(row):
            if row['true'] == 1 and row['pred'] == 1:
                return 'TP'
            elif row['true'] == 0 and row['pred'] == 0:
                return 'TN'
            elif row['true'] == 0 and row['pred'] == 1:
                return 'FP'
            else:
                return 'FN'
        
        results_df['result'] = results_df.apply(get_result, axis=1)
        
        # Sample stratification
        result_counts = results_df['result'].value_counts()
        sample_counts = {}
        
        for result, count in result_counts.items():
            proportion = count / len(results_df)
            sample_counts[result] = max(1, int(proportion * sample_size))
        
        # Adjust sample size to match target
        total = sum(sample_counts.values())
        if total < sample_size:
            majority_category = result_counts.index[0]
            sample_counts[majority_category] += (sample_size - total)
        elif total > sample_size:
            majority_category = result_counts.index[0]
            sample_counts[majority_category] -= (total - sample_size)
        
        # Create stratified sample
        sampled_dfs = []
        for result, sample_size in sample_counts.items():
            group_df = results_df[results_df['result'] == result]
            if len(group_df) <= sample_size:
                sampled_dfs.append(group_df)
            else:
                sampled_dfs.append(group_df.sample(n=sample_size, random_state=42))
        
        stratified_sample = pd.concat(sampled_dfs).reset_index(drop=True)
        
        # Scatter Plot
        markers = {'TP': '^', 'TN': 'o', 'FP': 's', 'FN': 'X'}
        colors = {'TP': 'darkgreen', 'TN': 'lightgreen', 'FP': 'darkred', 'FN': '#FF3333'}
        
        plt.figure(figsize=(12, 2))
        sns.set_style("whitegrid")
        
        for result, group in stratified_sample.groupby('result'):
            y_values = np.random.uniform(0.1, 0.9, size=len(group))
            plt.scatter(
                group['probability'],
                y_values,
                label=result,
                marker=markers[result],
                color=colors[result],
                alpha=0.7,
                s=80
            )
        
        legend_handles = [
            Line2D([0], [0], marker=markers[result], color='w', markerfacecolor=colors[result],
                  markersize=8, label=f"{result} (n={len(stratified_sample[stratified_sample['result'] == result])})")
            for result in ['TP', 'TN', 'FP', 'FN']
        ]
        
        threshold_pct = threshold * 100
        plt.axvline(x=threshold_pct, color='black', linestyle='--', linewidth=2.5, alpha=0.7, 
                   label=f'Decision Threshold ({threshold_pct:.0f}%)')
        plt.xlabel('Predicted Probability (%)', fontsize=12)
        plt.xlim(0, 100)
        plt.yticks([])
        plt.xticks(range(0, 101, 10))
        plt.legend(handles=legend_handles, title='Result Type', fontsize=10, loc='upper right')
        plt.tight_layout()
        plt.show()
        
        # Histogram
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 100, 21)
        
        for result, color in colors.items():
            subset = stratified_sample[stratified_sample['result'] == result]
            if len(subset) > 0:
                plt.hist(
                    subset['probability'],
                    bins=bins,
                    label=f"{result} (n={len(subset)})",
                    color=color,
                    edgecolor='black',
                    alpha=0.7
                )
        
        plt.axvline(x=threshold_pct, color='black', linestyle='--', linewidth=2.5, alpha=0.7, 
                   label=f'Decision Threshold ({threshold_pct:.0f}%)')
        plt.yscale('log')
        plt.xlabel('Predicted Probability (%)', fontsize=12)
        plt.ylabel('Number of Samples (Log Scale)', fontsize=12)
        plt.title('Distribution of Predicted Probabilities by Result Type', fontsize=14)
        plt.xlim(0, 100)
        plt.legend(title='Result Type', fontsize=10, loc='upper right')
        plt.tight_layout()
        plt.show()
        
        # Confusion Matrix & Classification Report
        conf_matrix = confusion_matrix(stratified_sample['true'], stratified_sample['pred'])
        print("\nConfusion Matrix (Stratified Sample):")
        print(conf_matrix)
        print("\nClassification Report (Stratified Sample):")
        print(classification_report(stratified_sample['true'], stratified_sample['pred']))


    def plot_calibration_curve(self, y_true, y_pred_proba, n_bins=10):
        """
        Plot the calibration curve for a model's predicted probabilities.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels.
        y_pred_proba : array-like
            Predicted probabilities for the positive class.
        n_bins : int, optional (default=10)
            Number of bins for calibration curve.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')

        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        
        plt.xlabel("Predicted Probability")
        plt.ylabel("Observed Frequency")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def evaluate_complete(self, model, X_test, y_test, threshold=0.5, figsize_cm=(20, 6), 
                        figsize_roc=(8, 8), figsize_threshold=(15, 5), sample_size=1500):
        """
        Run complete evaluation suite for classification model.
        
        Parameters:
        -----------
        model : estimator object
            Fitted model with predict_proba method.
        X_test : array-like
            Test features.
        y_test : array-like
            True binary labels.
        threshold : float, optional (default=0.5)
            Classification threshold for binary prediction.
        figsize_cm : tuple, optional (default=(20, 6))
            Figure size for confusion matrix plots.
        figsize_roc : tuple, optional (default=(8, 8))
            Figure size for ROC curve.
        figsize_threshold : tuple, optional (default=(15, 5))
            Figure size for threshold metric plots.
        sample_size : int, optional (default=1500)
            Number of points to include in the stratified visualization.
        """
        print("=" * 80)
        print("COMPLETE MODEL EVALUATION")
        print("=" * 80)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 1. Metrics and confusion matrix
        print("\n1. CLASSIFICATION METRICS AND CONFUSION MATRIX")
        self.evaluate_metrics(y_test, y_pred)
        
        # 2. ROC curve and AUC
        print("\n2. ROC CURVE AND AUC/GINI METRICS")
        self.plot_roc_curve(y_test, y_pred_proba, figsize=figsize_roc)
        
        # 4. Stratified visualization
        print("\n4. STRATIFIED CLASSIFICATION VISUALIZATION")
        self.stratified_classification_plot(model, X_test, y_test, threshold, sample_size)

        # 5. Calibration curve
        print("\n5. PRE-CALIBRATION MODEL")
        self.plot_calibration_curve(y_test, y_pred_proba)









# Plot recall precision that works for any other model than catboost 
# ============================================================
def plot_recall_precision_adapted(y_true, y_pred_proba, figsize=(15, 5), threshold_range=(0.01, 0.99, 0.01)):
    """
    Plot recall, precision, F1, F2, and F0.5 scores across different thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_pred_proba : array-like
        Probability estimates of the positive class.
    figsize : tuple, optional (default=(15, 5))
        Figure size (width, height) in inches.
    threshold_range : tuple, optional (default=(0.01, 0.99, 0.01))
        Range for threshold values as (start, stop, step).
    
    Returns:
    --------
    matplotlib.figure.Figure
        The matplotlib figure containing the plot.
    """
    # Convert inputs to numpy arrays to handle various input types
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Ensure no NaN values in predictions
    if np.isnan(y_pred_proba).any():
        y_pred_proba = np.nan_to_num(y_pred_proba)
    
    # Generate thresholds
    thresholds = np.arange(*threshold_range)
    
    # Initialize empty lists for metrics
    recalls = []
    precisions = []
    f1_scores = []
    f2_scores = []
    f05_scores = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        recalls.append(recall_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
        f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
        f05_scores.append(fbeta_score(y_true, y_pred, beta=0.5))
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot lines (no markers to avoid conversion issues)
    ax.plot(thresholds, recalls, '-', color="red", label='Recall', linewidth=2)
    ax.plot(thresholds, precisions, '-', color="blue", label='Precision', linewidth=2)
    ax.plot(thresholds, f1_scores, '-', color="limegreen", label='F1 Score', linewidth=2)
    ax.plot(thresholds, f2_scores, '-', color="gold", label='F2 Score', linewidth=2)
    ax.plot(thresholds, f05_scores, '-', color="pink", label='F0.5 Score', linewidth=2)
    
    # Find optimal F2 score directly from our calculated values
    best_idx = np.argmax(f2_scores)
    best_threshold = thresholds[best_idx]
    best_f2 = f2_scores[best_idx]
    
    # Highlight best F2 score
    ax.plot(best_threshold, best_f2, 'ko', markersize=10,
            label=f'Best F2 (th={best_threshold:.3f}, f2={best_f2:.3f})')
    
    # Set labels and formatting
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision-Recall Metrics vs Threshold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Set axis limits
    ax.set_xlim([min(thresholds), max(thresholds)])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    return fig





#==================================================================================
#============================== Calibrator ========================================
#==================================================================================
# Fitting all methods (7)


class Calibrator:
    """
    A class for calibrating and evaluating probabilistic classifiers.

    This class implements various calibration methods, evaluation metrics,
    and visualization tools for analyzing model calibration performance.

    Attributes:
        model: The base classifier to be calibrated
        calib_set: A tuple (X_calib, y_calib) for training calibrators
        test_set: A tuple (X_test, y_test) for evaluating calibrators
        calibrated_models: Dictionary to store fitted calibration models
        metrics: Dictionary to store evaluation metrics for each method
    """

    def __init__(self, model, calib_set=None, test_set=None):
        """
        Initialize the Calibrator.

        Args:
            model: Base classifier with predict_proba method
            calib_set: Tuple (X_calib, y_calib) for training calibrators
            test_set: Tuple (X_test, y_test) for evaluating calibrators
        """
        self.model = model
        self.calib_set = calib_set
        self.test_set = test_set
        self.calibrated_models = {}
        self.metrics = {}

    def fit_calibrator(self, method, **kwargs):
        """
        Fit a probability calibration model.

        Args:
            method: String specifying the calibration method
                   ('isotonic', 'sigmoid', 'exponential', 'polynomial',
                    'bbq', 'venn_abers', 'hist_binning')
            **kwargs: Additional parameters for specific calibration methods
                     - 'cv': Number of CV folds for isotonic/sigmoid
                     - 'degree': Polynomial degree
                     - 'n_bins', 'prior', 'max_samples': for BBQ
                     - 'cal_size', 'random_state': for Venn-Abers
                     - 'n_bins': for histogram binning
        """
        # ─── sklearn calibrators ───────────────────────────────────────────
        if method == 'isotonic':
            calibrated_clf = CalibratedClassifierCV(
                self.model,
                cv=kwargs.get('cv', 2),
                method="isotonic"
            )
            calibrated_clf.fit(*self.calib_set)

        elif method == 'sigmoid':
            calibrated_clf = CalibratedClassifierCV(
                self.model,
                cv=kwargs.get('cv', 2),
                method="sigmoid"
            )
            calibrated_clf.fit(*self.calib_set)

        # ─── custom curve‐fitting calibrators ─────────────────────────────
        elif method == 'exponential':
            Xc, yc = self.calib_set
            probs = self.model.predict_proba(Xc)[:, 1]
            calibrated_clf = ExponentialCalibration(eps=1e-15).fit(probs, yc)


        elif method == 'polynomial':
            degree = kwargs.get('degree', 3)
            # call the top‐level class, not self.
            calibrated_clf = PolynomialNonDecreasingCalibration(degree=degree)
            # pull out your calibration probs and labels
            Xc, yc = self.calib_set
            probs  = self.model.predict_proba(Xc)[:, 1]
            # fit the polynomial regressor
            calibrated_clf.fit(probs, yc)


        # ─── BBQ (Bayesian Binning into Quantiles) ─────────────────────────
        elif method == 'bbq':
            # 1) pull out raw calibration-set probabilities & labels
            Xc, yc = self.calib_set
            probs  = self.model.predict_proba(Xc)[:, 1]

            # 2) subsample down to max_samples if provided
            max_s = kwargs.get('max_samples', None)
            if max_s is not None and len(probs) > max_s:
                idx      = np.random.choice(len(probs), max_s, replace=False)
                probs, yc = probs[idx], yc[idx]

            # 3) instantiate & fit the new BBQCalibration
            calibrated_clf = BBQCalibration(
                n_bins_options=kwargs.get('n_bins_options', None),
                prior         = kwargs.get('prior', 'uniform'),
                gamma         = kwargs.get('gamma', 1.0)
            ).fit(probs, yc)

        # ─── Venn-Abers ─────────────────────────────────────────────────────
        elif method == 'venn_abers':
            cal_size     = kwargs.get('cal_size', 0.2)
            random_state = kwargs.get('random_state', 101)
            calibrated_clf = VennAbersCalibrator(
                estimator    = self.model,
                inductive    = True,
                cal_size     = cal_size,
                random_state = random_state
            )
            calibrated_clf.fit(*self.calib_set)

        elif method == 'hist_binning':
            # pull out the calibration set
            Xc, yc = self.calib_set

            try:
                requested_bins = kwargs.get('n_bins', 10)
                # cap by √N (minimum 2)
                max_bins = max(2, int(math.sqrt(len(yc))))
                n_bins   = min(requested_bins, max_bins)

                print(f"[hist_binning] calib set size = {len(yc)}, "
                    f"requested n_bins = {requested_bins}, using n_bins = {n_bins}")

                # instantiate & fit using the top-level class name
                calibrated_clf = HistBinningCalibration(n_bins=n_bins)

                probs = self.model.predict_proba(Xc)[:, 1]
                print(f"[hist_binning] raw probs shape = {probs.shape}, yc shape = {yc.shape}")

                calibrated_clf.fit(probs, yc)
                # store the fitted calibrator
                self.calibrated_models[method] = calibrated_clf
                print(f"Fitted hist_binning with {n_bins} bins")

            except Exception as e:
                logging.warning(
                    f"Calibration hist_binning failed "
                    f"(size={len(yc)}, ratio={(yc==1).mean():.4f}, bins={n_bins}): {e}"
                )


        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # ─── store & report ─────────────────────────────────────────────────
        self.calibrated_models[method] = calibrated_clf
        print(f"Fitted calibrator: {method}")

    def predict_proba(self, method: str) -> np.ndarray:
        """
        Get calibrated probabilities using the specified method.

        Args:
            method: String specifying the calibration method
                    ('uncalibrated', 'isotonic', 'sigmoid', etc.)

        Returns:
            numpy.ndarray: Calibrated probabilities for the positive class
        """
        X_test, _ = self.test_set

        # 1) Uncalibrated
        if method == 'uncalibrated':
            return self.model.predict_proba(X_test)[:, 1]

        # 2) Must have been fitted
        if method not in self.calibrated_models:
            raise ValueError(f"Calibrator for method '{method}' has not been fitted.")

        calib = self.calibrated_models[method]

        # 3) Methods that take raw features X
        if method in ['isotonic', 'sigmoid', 'venn_abers']:
            return calib.predict_proba(X_test)[:, 1]

        # 4) Methods that take raw probabilities
        raw_p = self.model.predict_proba(X_test)[:, 1]
        p     = np.clip(raw_p, 1e-15, 1 - 1e-15)

        if method == 'hist_binning':
            # use the clipped probs
            return calib.predict(p)

        elif method == 'bbq':
            return calib.predict(p)

        elif method in ['exponential', 'polynomial']:
            return calib.predict_proba(p)

        # 5) Fallback
        raise ValueError(f"Unknown calibration method: {method}")



    def evaluate(self, method, n_bins=15):
        """
        Evaluate a calibration method using various metrics.

        Args:
            method: String specifying the calibration method to evaluate
            n_bins: Number of bins to use for ECE calculation

        Returns:
            None. Metrics are stored in self.metrics
        """
        probs = self.predict_proba(method)
        brier = brier_score_loss(self.test_set[1], probs)
        logloss = log_loss(self.test_set[1], probs)
        ece = self.calculate_ece(self.test_set[1], probs, n_bins)
        self.metrics[method] = {
            'brier_score': brier,
            'log_loss': logloss,
            'ece': ece
        }
        print(f"Evaluated method: {method}")

    def calculate_ece(self, y_true, probs, n_bins=15):
        """
        Calculate Expected Calibration Error (ECE).

        Args:
            y_true: True binary labels
            probs: Predicted probabilities for the positive class
            n_bins: Number of bins to use

        Returns:
            float: The calculated ECE value
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges, right=True) - 1
        ece = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                bin_prob = np.mean(probs[mask])
                bin_true = np.mean(y_true[mask])
                ece += np.abs(bin_true - bin_prob) * np.sum(mask)
        ece /= len(y_true)
        return ece

    def plot_calibration_curves(self, methods=None, n_bins=15):
        """
        Plot calibration curves for specified methods.

        Args:
            methods: List of calibration methods to plot
            n_bins: Number of bins to use for plotting

        Returns:
            None. Displays the plot.
        """
        if methods is None:
            methods = list(self.calibrated_models.keys()) + ['uncalibrated']

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.get_cmap("Dark2")
        markers = ["x", "s", "o", "^", "D", "*"]

        for i, method in enumerate(methods):
            probs = self.predict_proba(method)
            display = CalibrationDisplay.from_predictions(
                self.test_set[1],
                probs,
                n_bins=n_bins,
                ax=ax,
                name=method,
                color=colors(i),
                marker=markers[i % len(markers)],
            )

        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Curve Comparison')
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def print_metrics(self):
        """
        Print evaluation metrics for all calibration methods.

        Returns:
            None. Prints metrics to console.
        """
        print("\nMetrics:")
        print("--------")
        for method, metrics in self.metrics.items():
            print(f"{method}:")
            print(f"  Brier Score: {metrics['brier_score']:.6f}")
            print(f"  Log Loss: {metrics['log_loss']:.6f}")
            print(f"  ECE: {metrics['ece']:.6f}")

    def summary_table(self):
        """
        Create a summary table of all metrics for all methods.

        Returns:
            pandas.DataFrame: Summary table with metrics for each method
        """
        summary = []
        for method, metrics in self.metrics.items():
            summary.append({
                'Method': method,
                'Brier Score': metrics['brier_score'],
                'Log Loss': metrics['log_loss'],
                'ECE': metrics['ece']
            })
        df = pd.DataFrame(summary)
        return df

    def summary_table_significant_figures(self, decimals=3):
        """
        Create a summary table with rounded metrics.

        Args:
            decimals: Number of decimal places to round to

        Returns:
            pandas.DataFrame: Summary table with rounded metrics
        """
        summary = []
        for method, metrics in self.metrics.items():
            summary.append({
                'Method': method,
                'Brier Score': round(metrics['brier_score'], decimals),
                'Log Loss': round(metrics['log_loss'], decimals),
                'ECE': round(metrics['ece'], decimals)
            })
        df = pd.DataFrame(summary)
        return df

    def summary_table_scientific_notation(self):
        """
        Create a summary table with metrics in scientific notation.

        Returns:
            pandas.DataFrame: Summary table with metrics in scientific notation
        """
        summary = []
        for method, metrics in self.metrics.items():
            summary.append({
                'Method': method,
                'Brier Score': f"{metrics['brier_score']:.1e}",
                'Log Loss': f"{metrics['log_loss']:.1e}",
                'ECE': f"{metrics['ece']:.1e}"
            })
        df = pd.DataFrame(summary)
        return df

    def best_method(self, metric='average'):
        """
        Determine the best calibration method based on specified metrics.
        
        Args:
            metric: String specifying which metric to use
                ('brier_score', 'log_loss', 'ece', or 'average')
                If 'average' is specified, all metrics are averaged.
        
        Returns:
            str: Name of the best method
        """
        if metric == 'average':
            # Calculate the average score for each method across all metrics
            avg_scores = {}
            for method in self.metrics:
                # Calculate the average of all metrics
                avg_scores[method] = (
                    self.metrics[method]['brier_score'] + 
                    self.metrics[method]['log_loss'] + 
                    self.metrics[method]['ece']
                ) / 3
            
            # Find the method with the lowest average score
            best_method = min(avg_scores, key=avg_scores.get)
        else:
            # Original behavior for individual metrics
            if metric not in ['brier_score', 'log_loss', 'ece']:
                raise ValueError(f"Unknown metric: {metric}")
                
            best_method = min(self.metrics, key=lambda method: self.metrics[method][metric])
        
        return best_method

    def plot_histograms(self, method='isotonic', optimal_threshold=0.075):
        """
        Plot histograms of uncalibrated and calibrated probabilities by class.

        Args:
            method: String specifying which calibration method to visualize
            optimal_threshold: Threshold value to highlight in the plot

        Returns:
            None. Displays the plot.
        """
        # Get the probabilities for uncalibrated and calibrated methods
        prob_predictions = self.predict_proba('uncalibrated')
        probs_cal_isotonic = self.predict_proba(method)

        # Separate probabilities by class
        minority_class_0 = prob_predictions[self.test_set[1] == 0]
        minority_class_1 = prob_predictions[self.test_set[1] == 1]

        minority_class_0_cal = probs_cal_isotonic[self.test_set[1] == 0]
        minority_class_1_cal = probs_cal_isotonic[self.test_set[1] == 1]

        # Debugging: Print the shapes and some values of the arrays
        print(f"Uncalibrated Class 0: {minority_class_0.shape}, {minority_class_0[:5]}")
        print(f"Uncalibrated Class 1: {minority_class_1.shape}, {minority_class_1[:5]}")
        print(f"Calibrated Class 0: {minority_class_0_cal.shape}, {minority_class_0_cal[:5]}")
        print(f"Calibrated Class 1: {minority_class_1_cal.shape}, {minority_class_1_cal[:5]}")

        # Check if there are any samples for class 1
        if minority_class_1.size == 0 or minority_class_1_cal.size == 0:
            print("Warning: No samples for class 1 in the test set.")
            return

        # Compute histograms
        counts_0, bins_0 = np.histogram(minority_class_0, bins=20)
        counts_1, bins_1 = np.histogram(minority_class_1, bins=20)

        counts_0_cal, bins_0_cal = np.histogram(minority_class_0_cal, bins=20)
        counts_1_cal, bins_1_cal = np.histogram(minority_class_1_cal, bins=20)

        # Debugging: Print the counts and bins
        print(f"Counts Class 0: {counts_0}")
        print(f"Bins Class 0: {bins_0}")
        print(f"Counts Class 1: {counts_1}")
        print(f"Bins Class 1: {bins_1}")
        print(f"Counts Class 0 Calibrated: {counts_0_cal}")
        print(f"Bins Class 0 Calibrated: {bins_0_cal}")
        print(f"Counts Class 1 Calibrated: {counts_1_cal}")
        print(f"Bins Class 1 Calibrated: {bins_1_cal}")

        max_count = max(counts_0.max(), counts_1.max(), counts_0_cal.max(), counts_1_cal.max())

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.histplot(minority_class_0, bins=20, color='blue', label="Probabilities Class 0")
        plt.axvline(0.5, color='red', linestyle='--', label="Default Threshold (0.5)")
        plt.yscale('log')
        plt.ylim(0, max_count * 10)  # Set the same log scale for all plots
        plt.xlim(0, 1)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency (log)")
        plt.legend()
        plt.title("Uncalibrated - Class 0")

        plt.subplot(2, 2, 2)
        sns.histplot(minority_class_1, bins=20, color='orange', label="Probabilities Class 1")
        plt.axvline(0.5, color='red', linestyle='--', label="Default Threshold (0.5)") # Set the same log scale for all plots
        plt.xlim(0, 1)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency ")
        plt.legend()
        plt.title("Uncalibrated - Class 1")

        plt.subplot(2, 2, 3)
        sns.histplot(minority_class_0_cal, bins=20, color='blue', label="Probabilities Class 0")
        plt.axvline(0.5, color='red', linestyle='--', label="Default Threshold (0.5)")
        plt.axvline(optimal_threshold, color='green', linestyle='--', label="Optimal Threshold")
        plt.yscale('log')
        plt.ylim(0, max_count * 10)  # Set the same log scale for all plots
        plt.xlim(0, 1)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency (log)")
        plt.legend()
        plt.title(f"Calibrated {method.upper()} - Class 0")

        plt.subplot(2, 2, 4)
        sns.histplot(minority_class_1_cal, bins=20, color='orange', label="Probabilities Class 1")
        plt.axvline(0.5, color='red', linestyle='--', label="Default Threshold (0.5)")
        plt.axvline(optimal_threshold, color='green', linestyle='--', label="Optimal Threshold")
  # Set the same log scale for all plots
        plt.xlim(0, 1)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title(f"Calibrated {method.upper()} - Class 1")

        plt.tight_layout()
        plt.show()

    def print_probability_counts(self, method='isotonic', threshold=0.5):
        """
        Print counts of probabilities above and below a specified threshold for both uncalibrated and calibrated methods.

        Args:
            method: String specifying which calibration method to analyze.
                    Options include 'uncalibrated', 'isotonic', 'sigmoid',
                    'exponential', 'polynomial', 'bbq', 'venn_abers'.
            threshold: The probability threshold to use for counting.

        Returns:
            None. Prints counts to console.
        """
        # Get the uncalibrated probabilities
        uncalibrated_probs = self.model.predict_proba(self.test_set[0])[:, 1]

        # Separate uncalibrated probabilities by class
        uncalibrated_probs_class0 = uncalibrated_probs[self.test_set[1] == 0]
        uncalibrated_probs_class1 = uncalibrated_probs[self.test_set[1] == 1]

        # Count uncalibrated probabilities above and below the threshold
        uncalibrated_above_threshold_class0 = np.sum(uncalibrated_probs_class0 > threshold)
        uncalibrated_below_threshold_class0 = np.sum(uncalibrated_probs_class0 <= threshold)
        uncalibrated_above_threshold_class1 = np.sum(uncalibrated_probs_class1 > threshold)
        uncalibrated_below_threshold_class1 = np.sum(uncalibrated_probs_class1 <= threshold)

        # Print the uncalibrated counts
        print(f"Uncalibrated - Class 0: {uncalibrated_above_threshold_class0} probabilities > {threshold}, {uncalibrated_below_threshold_class0} probabilities <= {threshold}")
        print(f"Uncalibrated - Class 1: {uncalibrated_above_threshold_class1} probabilities > {threshold}, {uncalibrated_below_threshold_class1} probabilities <= {threshold}")

        # Get the calibrated probabilities for the specified method
        if method == 'uncalibrated':
            calibrated_probs = uncalibrated_probs
        else:
            calibrated_probs = self.predict_proba(method)

        # Separate calibrated probabilities by class
        calibrated_probs_class0 = calibrated_probs[self.test_set[1] == 0]
        calibrated_probs_class1 = calibrated_probs[self.test_set[1] == 1]

        # Count calibrated probabilities above and below the threshold
        calibrated_above_threshold_class0 = np.sum(calibrated_probs_class0 > threshold)
        calibrated_below_threshold_class0 = np.sum(calibrated_probs_class0 <= threshold)
        calibrated_above_threshold_class1 = np.sum(calibrated_probs_class1 > threshold)
        calibrated_below_threshold_class1 = np.sum(calibrated_probs_class1 <= threshold)

        # Print the calibrated counts
        print(f"{method.capitalize()} - Class 0: {calibrated_above_threshold_class0} probabilities > {threshold}, {calibrated_below_threshold_class0} probabilities <= {threshold}")
        print(f"{method.capitalize()} - Class 1: {calibrated_above_threshold_class1} probabilities > {threshold}, {calibrated_below_threshold_class1} probabilities <= {threshold}")

    def consistent_probability_analysis(self, method='isotonic', n_bins=10, optimal_threshold=None):
        """
        Analyze probabilities with consistent binning between histogram and counts.

        Parameters:
        -----------
        method : str, default='isotonic'
            The calibration method to analyze.
        n_bins : int, default=10
            Number of bins to use in the histogram.
        optimal_threshold : float, optional
            If provided, creates special binning with this threshold.

        Returns:
        --------
        tuple
            (counts, bins) - The counts per bin and the bin edges.
        """
        # Get the probabilities for the specified method
        if method == 'uncalibrated':
            probs = self.model.predict_proba(self.test_set[0])[:, 1]
        else:
            probs = self.predict_proba(method)

        # Separate by class for detailed analysis
        probs_class0 = probs[self.test_set[1] == 0]
        probs_class1 = probs[self.test_set[1] == 1]

        # Create consistent bins
        if optimal_threshold is not None and 0 < optimal_threshold < 1:
            # Create bins with special focus on optimal threshold
            if optimal_threshold > 0.1:
                # Make sure we have bins below the threshold
                below_bins = np.linspace(0, optimal_threshold, max(2, int(n_bins * optimal_threshold)))
                above_bins = np.linspace(optimal_threshold, 1, max(2, int(n_bins * (1 - optimal_threshold))))
                bins = np.unique(np.concatenate([below_bins, above_bins]))
            else:
                # For very small thresholds, ensure we have enough resolution
                small_bins = np.linspace(0, 2*optimal_threshold, 4)
                main_bins = np.linspace(2*optimal_threshold, 1, n_bins - 2)
                bins = np.unique(np.concatenate([small_bins, main_bins]))
        else:
            # Create equal-width bins
            bins = np.linspace(0, 1, n_bins + 1)

        # Count probabilities in each bin (overall)
        counts = np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            counts[i] = np.sum((probs >= bins[i]) & (probs < bins[i+1]))

        # Count probabilities in each bin (by class)
        counts_class0 = np.zeros(len(bins) - 1)
        counts_class1 = np.zeros(len(bins) - 1)

        for i in range(len(bins) - 1):
            counts_class0[i] = np.sum((probs_class0 >= bins[i]) & (probs_class0 < bins[i+1]))
            counts_class1[i] = np.sum((probs_class1 >= bins[i]) & (probs_class1 < bins[i+1]))

        # Add edge case for values exactly equal to 1.0
        if bins[-1] == 1.0:
            counts[-1] += np.sum(probs == 1.0)
            counts_class0[-1] += np.sum(probs_class0 == 1.0)
            counts_class1[-1] += np.sum(probs_class1 == 1.0)

        # Print overall counts
        print(f"\nProbability distribution for {method} calibration:")
        print("-" * 60)
        print(f"{'Bin Range':<20} {'Total':<10} {'Class 0':<10} {'Class 1':<10}")
        print("-" * 60)
        total_count = 0

        for i in range(len(bins) - 1):
            bin_range = f"{bins[i]:.3f} to {bins[i+1]:.3f}"
            print(f"{bin_range:<20} {int(counts[i]):<10} {int(counts_class0[i]):<10} {int(counts_class1[i]):<10}")
            total_count += counts[i]

        print("-" * 60)
        print(f"{'Total':<20} {int(total_count):<10} {int(np.sum(counts_class0)):<10} {int(np.sum(counts_class1)):<10}")

        # Also print the simpler below/above threshold counts
        if optimal_threshold is not None:
            below_count = np.sum(probs < optimal_threshold)
            above_count = np.sum(probs >= optimal_threshold)
            below_count_class0 = np.sum(probs_class0 < optimal_threshold)
            below_count_class1 = np.sum(probs_class1 < optimal_threshold)
            above_count_class0 = np.sum(probs_class0 >= optimal_threshold)
            above_count_class1 = np.sum(probs_class1 >= optimal_threshold)

            print(f"\nUsing optimal threshold {optimal_threshold}:")
            print("-" * 60)
            print(f"{'Range':<20} {'Total':<10} {'Class 0':<10} {'Class 1':<10}")
            print("-" * 60)
            print(f"{'Below ' + str(optimal_threshold):<20} {below_count:<10} {below_count_class0:<10} {below_count_class1:<10}")
            print(f"{'Above ' + str(optimal_threshold):<20} {above_count:<10} {above_count_class0:<10} {above_count_class1:<10}")
            print("-" * 60)
            print(f"{'Total':<20} {below_count + above_count:<10} {below_count_class0 + above_count_class0:<10} {below_count_class1 + above_count_class1:<10}")

        # Compare with print_probability_counts values
        print("\nFor comparison with print_probability_counts:")
        below_05_class0 = np.sum(probs_class0 <= 0.5)
        above_05_class0 = np.sum(probs_class0 > 0.5)
        below_05_class1 = np.sum(probs_class1 <= 0.5)
        above_05_class1 = np.sum(probs_class1 > 0.5)

        print(f"{method.capitalize()} - Class 0: {above_05_class0} probabilities > 0.5, {below_05_class0} probabilities <= 0.5")
        print(f"{method.capitalize()} - Class 1: {above_05_class1} probabilities > 0.5, {below_05_class1} probabilities <= 0.5")

        # Plot histogram with the same bins (separate plots for overall and by class)
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        # First plot: Overall distribution
        axs[0].hist(probs, bins=bins, alpha=0.7, color='blue', label='All samples')
        axs[0].set_title(f"Overall Probability Distribution for {method} Calibration")
        axs[0].set_xlabel("Predicted Probability")
        axs[0].set_ylabel("Count")

        # Highlight thresholds
        axs[0].axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
        if optimal_threshold is not None:
            axs[0].axvline(x=optimal_threshold, color='green', linestyle='--',
                          label=f'Optimal Threshold ({optimal_threshold})')
        axs[0].legend()

        # Second plot: Distribution by class
        axs[1].hist(probs_class0, bins=bins, alpha=0.7, color='blue', label='Class 0')
        axs[1].hist(probs_class1, bins=bins, alpha=0.7, color='orange', label='Class 1')
        axs[1].set_title(f"Probability Distribution by Class for {method} Calibration")
        axs[1].set_xlabel("Predicted Probability")
        axs[1].set_ylabel("Count")

        # Highlight thresholds
        axs[1].axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
        if optimal_threshold is not None:
            axs[1].axvline(x=optimal_threshold, color='green', linestyle='--',
                         label=f'Optimal Threshold ({optimal_threshold})')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

        return counts, bins