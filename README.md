Decision-Focused Learning for Multi-Task Prediction and Integrated Energy System Operation
Overview
This repository contains the implementation of a multi-task decision-focused learning framework for integrated energy system (IES) operation under multiple uncertainties.
The proposed framework establishes a closed-loop connection between prediction and operational decision-making. Rather than optimizing prediction accuracy alone, it incorporates the operational consequences of prediction errors into the learning process, allowing the predictive model to better account for their effects on system scheduling and operating costs.
The framework jointly considers electricity-load and photovoltaic (PV) generation forecasting and integrates prediction outputs with day-ahead optimization and intraday scheduling.
Methodology
The framework consists of the following main components:
Multi-task prediction: Joint forecasting of electricity load and PV generation.
Decision-loss characterization: Modeling the operational consequences of prediction errors, including their magnitude, direction, temporal characteristics, and interactions between uncertain sources.
Decision-loss approximation: Fitting surrogate functions to represent the relationship between prediction errors and operational losses.
Decision-feedback learning: Incorporating decision-related loss information into model training to improve the operational relevance of predictions.
IES scheduling and evaluation: Evaluating the effects of prediction outputs on day-ahead and intraday operational decisions and system operating costs.
Repository Structure
The repository contains Python source files implementing the prediction, decision-loss modeling, and integrated energy system scheduling components.
The specific functions and dependencies of individual scripts should be identified from their source code and execution workflow.
Requirements
The computational environment used in this study is based on Python 3.12.
Additional Python packages and optimization solvers may be required, depending on the selected scripts and experimental configuration. The required dependencies and solver settings should be configured before running the corresponding modules.
Data Availability
The industrial electricity-load and PV generation data used in the case study are not publicly available at present. Consequently, the original case-study data are not included in this repository.
Meteorological data used in the study were obtained from the Open-Meteo historical weather database:
https://open-meteo.com/
Users seeking to reproduce the experiments should ensure that they have access to the required input data and configure the data paths and model parameters accordingly.
Reproducibility
The source code is provided to document the computational implementation of the proposed framework.
Reproducing the reported results requires the appropriate input data, model configurations, software dependencies, and optimization solver settings. Numerical results may vary depending on the computational environment and solver configuration.
Citation
If you use this code in your research, please cite the associated research article:
The bibliographic information will be added after publication.
License
A license has not yet been specified. The terms governing the use, modification, and redistribution of this code will be provided when a license is selected.
