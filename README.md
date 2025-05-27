# Calibration 

This repository contains the `CalibrationFunctions.py` module, which provides a modular framework for preprocessing data, calibrating probabilistic classifiers, and evaluating model performance.

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
