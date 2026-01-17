# MulU-AP

## Overview
This project implemented experiments related to the manuscript "Synergistic Estimation of Six Air Pollutant Concentrations Using a Multitask Network with Uncertain Losses: A Case Study of Sichuan Province, China". The project aims to proposed a multitask neural network to collaboratively estimate seamless six common atmospheric pollutants (PM10, PM2.5, O₃, CO, NO₂, and SO₂) concentration from remote sensing data.


## Structure
The main program includes six Python (.py) files:
pollution_multitask.py, pollution_multitask_predcsv.py, pollution_multitask_predcsv_plot.py,
pollution_multitask_shap.py, pollution_multitask_shap_importance.py, and pollution_multitask_shap_scatter.py.They correspond to Python scripts for multitask neural network training, prediction, visualization, SHAP-based interpretation, and SHAP feature-importance visualization, respectively.

## Usage
The pollution_multitask.py script operates based on a CSV file that provides the required input features, with detailed descriptions and configurations defined in the Python file.

## Environment
- Python 3.12
- PyTorch 2.6.0
  

## Contact
Author：Yunhui Tan
Email：tanyunhui@cug.edu.cn
