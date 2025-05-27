# Calibration 

This repository implements a comprehensive benchmarking framework for handling imbalanced binary classification problems and improving probability estimates through calibration. At its core is the `CalibrationFunctions.py` module, which offers a flexible, plug-and-play interface for:

- **Preprocessing & sampling** (oversampling, undersampling, cost-sensitive splits)  
- **Probabilistic calibration** (Platt scaling, isotonic regression, exponential, polynomial, histogram binning, BBQ, and more)  
- **Performance evaluation** (discrimination metrics like ROC-AUC and precision/recall, plus calibration metrics like Brier score and calibration curves)

Complementing this, the `Loop.ipynb` notebook walks through an end-to-end scenario: generating multiple imbalance and sampling configurations, fitting baseline and calibrated models, and systematically comparing their effectiveness. The primary aim of this study is to quantify how different combinations of imbalance-handling techniques and calibration methods interact, and to identify the conditions under which calibration yields the greatest benefit. 

## Table of Contents

* [Features](#features)
* [Dependencies](#dependencies)
* [License](#license)

## Features

* **DataPreprocessor** for loading, sampling, and splitting datasets into training, test, and calibration sets, with support for undersampling or custom imbalance ratios 
* **Calibrator** wrapper for fitting and evaluating multiple calibration methods (isotonic, platt, exponential, polynomial, BBQ, Venn–Abers, histogram binning) and computing metrics such as ECE, Brier score, and log loss
* **Individual Calibrators** (e.g., `ExponentialCalibration`, `PolynomialNonDecreasingCalibration`, `HistBinningCalibration`, `BBQCalibration`) for direct use in custom workflows
* **ModelEvaluator** class for computing classification metrics, plotting confusion matrices, ROC curves, and precision–recall analyses


## Dependencies

Listed in `requirements.txt`

## License

This project is licensed under the MIT License.
