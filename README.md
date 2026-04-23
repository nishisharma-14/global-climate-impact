# Global Warming Data Analysis Project

This project explores long-term climate trends using a global warming dataset and applies:

- data cleaning and preprocessing
- exploratory data analysis (EDA)
- statistical testing
- probability distribution analysis
- machine learning for temperature prediction
- policy and urbanization impact analysis

The repository contains two runnable Python scripts that cover similar workflows at different levels of detail.

## Project Goals

The analysis is designed around practical climate questions:

1. How are temperature anomalies and emissions changing over time?
2. Which climate indicators are most correlated with temperature change?
3. Did CO2 emissions significantly change after 1980?
4. Can average temperature be predicted from emission and policy-related variables?
5. How do policy scores vary across emission categories?
6. How does urbanization relate to environmental impact indicators?

## Repository Structure

- `global_warming_dataset.csv` - Source dataset used in all analyses.
- `global_warming_project.py` - Main, well-structured end-to-end analysis pipeline.
- `INTPRO.py` - Extended version with additional objective blocks (including urbanization analysis).
- `README.md` - Project documentation.

## Tech Stack

- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn

## Installation

1. Clone or download this repository.
2. Open the project folder in VS Code (or your terminal).
3. Install required packages:   

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## How To Run

Run either script from the project root:

```bash
python global_warming_project.py
```

or

```bash
python INTPRO.py
```

Both scripts will:

- print analysis output in the terminal
- generate multiple visualizations (line plots, histograms, heatmaps, boxplots, and model diagnostics)

## Key Analysis Components

### 1. Data Loading and Preparation
- Loads climate data from CSV
- Checks data types and missing values
- Creates helper features such as era/decade and emission categories

### 2. Visualization
- Temperature anomaly trend over years
- CO2 emissions distribution
- Feature correlation heatmap
- Emission-category boxplots

### 3. Statistical and Distribution Analysis
- Summary statistics, skewness, covariance
- Outlier detection using IQR fences
- Independent two-sample t-test (Pre-1980 vs Post-1980 CO2)
- Normal distribution fit for temperature anomaly

### 4. Machine Learning
- Linear Regression model to predict `Average_Temperature`
- Train/test split with feature scaling
- Evaluation via MSE, RMSE, and R2
- Coefficient interpretation for climate features

### 5. Policy and Urbanization Insights
- Policy score comparison across emission groups
- Urbanization level grouping and impact comparison

## Notes

- Make sure `global_warming_dataset.csv` is present in the same folder as the scripts before running.
- If plots do not appear, verify that your Python environment supports GUI plotting (or run in an environment like VS Code with plotting support).

## Future Improvements

- Add a `requirements.txt` file for reproducible setup
- Add model comparison (Random Forest, XGBoost, etc.)
- Include time-series forecasting methods
- Export charts/results to a report folder automatically

## Author

Created as a climate analytics and machine learning project for academic and learning purposes.
