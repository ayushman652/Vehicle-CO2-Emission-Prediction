# Multiple Linear Regression — Vehicle CO₂ Emission Prediction

A learning-focused machine learning project that predicts a vehicle's CO₂ emissions from **engine size** and **combined fuel economy (MPG)**. It extends a simple linear regression exercise by using two predictors, exploring feature relationships, standardizing inputs, fitting an ordinary least-squares model, and evaluating predictions on held-out data.

This repository uses small, separate Python modules so each stage of the ML workflow can be understood independently.

## Results

On an 80/20 train/test split (`random_state=42`), the trained two-feature model produced:

| Metric | Test result |
|---|---:|
| Mean absolute error (MAE) | 14.29 g/km |
| Mean squared error (MSE) | 466.11 (g/km)² |
| Root mean squared error (RMSE) | 21.59 g/km |
| R² | 0.8873 |

R² = 0.8873 means that the model accounts for about **88.73% of the variation in CO₂ emissions in this test set**, relative to predicting the test-set mean. It does not mean that 88.73% of individual predictions are correct.

The model's equation, expressed in the **original input units**, is approximately:

\[
\widehat{\mathrm{CO_2}} = 329.1364 + 17.8581\,\mathrm{ENGINESIZE} - 5.0150\,\mathrm{FUELCONSUMPTION\_COMB\_MPG}
\]

CO₂ emissions are in g/km, engine size is in litres, and combined fuel economy is in miles per gallon (MPG). These coefficients describe associations **conditional on the other included feature**, not causal effects. The intercept corresponds to zero engine size and zero MPG and has little practical interpretation.

## Dataset and problem

The project uses the **FuelConsumptionCo2.csv** dataset from IBM's machine learning course. It contains **1,067 rows and 13 columns** of vehicle information for model year 2014. The prediction target is `CO2EMISSIONS` (g/km).

**Selected predictors**

- `ENGINESIZE`: engine displacement in litres.
- `FUELCONSUMPTION_COMB_MPG`: combined city/highway fuel economy in MPG. Higher MPG generally means less fuel used per distance travelled.

**Target:** `CO2EMISSIONS` — CO₂ emitted per kilometre, in g/km.

Dataset source: [IBM FuelConsumptionCo2.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv)

The dataset is not committed to the repository in the shared-workspace setup. Download it from the link above and place it at the path described in **Setup**.

## Project structure

```text
AI-Engineering-workspace/
├── .venv/                             # Shared local virtual environment; not committed
├── datasets/
│   └── co2/
│       └── FuelConsumptionCo2.csv     # Shared local dataset; not committed
└── 02-Multiple-Linear-Regression/     # This repository
    ├── main.py
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data_loader.py
    │   ├── eda.py
    │   ├── preprocessing.py
    │   ├── trainer.py
    │   ├── evaluator.py
    │   └── visualizer.py
    ├── outputs/
    │   ├── correlation_heatmap.png
    │   └── actual_vs_predicted.png
    ├── requirements.txt
    ├── .gitignore
    └── README.md
```

The directory above shows the intended repository layout. `requirements.txt`, `.gitignore`, and this README are the final repository-preparation files; create or copy them into the project if not already present. The `eda.py` function can be called separately from `main.py` when the correlation heatmap needs to be regenerated.

## How the project works

1. **Load:** Read the vehicle dataset into a pandas DataFrame.
2. **Explore:** Calculate correlations between numeric columns and generate a heatmap.
3. **Prepare:** Select engine size and combined MPG, split the observations into training and testing subsets, and standardize the inputs using training statistics only.
4. **Train:** Fit a two-feature `LinearRegression` model on the standardized training inputs.
5. **Interpret:** Inspect standardized coefficients and convert them back into the original feature units.
6. **Evaluate:** Predict CO₂ emissions for the untouched test set and calculate MAE, MSE, RMSE, and R².
7. **Visualize:** Compare actual and predicted test emissions in a scatter plot with a perfect-prediction reference line.

## Module-by-module explanation: what, why, and how

### `src/config.py` — File paths

**What:** Defines the project root, shared workspace root, dataset path, and output directory using `pathlib.Path`.

**Why:** Hard-coding absolute Windows paths throughout the code would make the project difficult to move or share. A single configuration module keeps paths consistent.

**How:** `Path(__file__).resolve()` locates the configuration file; `.parent` moves up the directory tree. `OUTPUTS_DIR` points to the folder where generated figures are saved.

**Important:** This version expects the dataset in a *sibling* `datasets/co2/` folder one level above the project. For a standalone checkout, either recreate that layout or change `DATASET_PATH` in `config.py`.

### `src/data_loader.py` — Reading the dataset

**What:** `load_data()` calls `pd.read_csv(DATASET_PATH)` and returns a pandas DataFrame.

**Why:** Keeping file I/O separate means preprocessing and training functions can accept an already-loaded DataFrame without knowing where the CSV is stored.

**How:** pandas reads the CSV header into column names and each subsequent record into a row. The result can be inspected with `df.shape`, `df.head()`, `df.info()`, and `df.isna().sum()`.

### `src/eda.py` — Exploratory data analysis and correlation

**What:** `analyze_correlation(dataframe)` selects numeric columns, calculates their Pearson correlation matrix with `DataFrame.corr()`, prints it, and draws a seaborn heatmap.

**Why:** EDA reveals relationships between features and the target, possible redundancy between predictors, and data properties worth investigating before fitting a model.

**How Pearson correlation works:** For two numeric variables, X and Y,

\[
r_{XY}=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
\]

The numerator measures whether X and Y vary together; the denominator scales the result to the interval −1 to +1. Values near +1 indicate a strong positive *linear* relationship, values near −1 a strong negative one, and values near 0 little linear association. **Correlation does not establish causation.**

**Observed correlations in this dataset** (rounded):

| Feature pair | Pearson r | Interpretation |
|---|---:|---|
| Engine size ↔ CO₂ | +0.87 | Larger engines tend to have higher emissions. |
| Combined MPG ↔ CO₂ | −0.91 | Higher MPG tends to accompany lower emissions. |
| Engine size ↔ cylinders | +0.93 | These predictors carry overlapping information. |
| City ↔ combined fuel consumption | ≈ +1.00 | These measurements are highly redundant. |

`MODELYEAR` is constant at 2014 in this dataset, so its standard deviation is zero and its Pearson correlations are undefined (`NaN`).

**Heatmap:** `sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")` maps correlation values to colours and writes their rounded values into the cells. A heatmap is a *visual representation of the correlation matrix*, not a separate statistical test.

**Multicollinearity:** Highly correlated predictors can make individual regression coefficients harder to interpret or less stable. For this learning project, we use engine size and combined MPG instead of including every highly related fuel-consumption measurement.

### `src/preprocessing.py` — Feature selection, splitting, and scaling

**What:** `preprocess_data(dataframe)` selects `ENGINESIZE` and `FUELCONSUMPTION_COMB_MPG` as X and `CO2EMISSIONS` as y, splits observations 80/20 using `train_test_split(test_size=0.2, random_state=42)`, and standardizes the two input columns with `StandardScaler`.

**Why split:** The model learns from the training set; the held-out test set provides an estimate of performance on unseen observations. The fixed random seed makes the random split reproducible.

**Why standardize:** Engine size and MPG use different units and ranges. Standardization places both on a mean-zero, unit-variance scale. It is useful for understanding preprocessing and standardized coefficients, although **ordinary least-squares linear regression with an intercept does not require scaling** and typically yields the same predictions without it, apart from numerical rounding.

**Mathematics of standardization:**

\[
z_j=\frac{x_j-\mu_j}{\sigma_j}
\]

Here, \(\mu_j\) and \(\sigma_j\) are the **training-set** mean and standard deviation of feature j. Scikit-learn's `StandardScaler` uses the population-standard-deviation convention (`ddof=0`).

**The three methods:**

- `scaler.fit(X_train)`: learn and store training-feature means (`mean_`) and scaling factors (`scale_`); return the fitted scaler.
- `scaler.transform(X)`: use those *stored* values to transform any compatible feature matrix; do not relearn statistics.
- `scaler.fit_transform(X_train)`: fit and transform the training set in one call.

**Correct sequence:**

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why not `fit_transform(X_test)`?** Fitting the scaler on the test set would let test-set information influence preprocessing and put the two datasets on different scales. This is **data leakage**. The same fitted scaler must also be used to transform new vehicles at prediction time.

**Observed verification:** The split produced 853 training rows and 214 testing rows, each with two features. The standardized training means were approximately 0 (`1.27e-16` and `6.87e-17`, numerical rounding) and both standard deviations were 1.

### `src/trainer.py` — Ordinary least-squares regression

**What:** `train_model(X_train, y_train)` creates `LinearRegression()`, calls `model.fit(X_train, y_train)`, and returns the fitted model.

**Why:** Multiple linear regression models a continuous target using a weighted sum of multiple features. It is an interpretable baseline for tabular regression problems.

**Model:**

\[
\hat y_i=b+w_1z_{i1}+w_2z_{i2}
\]

The weights are selected by **ordinary least squares (OLS)**, which minimizes the sum of squared residuals:

\[
\min_{b,w_1,w_2}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

A residual is the actual value minus its prediction. Squaring prevents positive and negative residuals from cancelling and penalizes larger errors more heavily.

**Fitted standardized coefficients:**

| Parameter | Value |
|---|---:|
| Engine size | +25.2492 |
| Combined MPG | −36.6058 |
| Intercept | 257.2567 |

The intercept is the predicted CO₂ emissions when both standardized inputs are zero—equivalently, when both raw features equal their training-set means. Each standardized coefficient describes the predicted change in emissions associated with a one-training-standard-deviation increase in that feature, **holding the other feature constant**.

**Converting coefficients to original units:** Since \(z_j=(x_j-\mu_j)/\sigma_j\), substitute this expression into the model and collect terms:

\[
w_j^{\mathrm{original}}=\frac{w_j^{\mathrm{standardized}}}{\sigma_j}
\]

\[
b^{\mathrm{original}}=b^{\mathrm{standardized}}-\sum_jw_j^{\mathrm{original}}\mu_j
\]

In Python:

```python
original_coefficients = model.coef_ / scaler.scale_
original_intercept = model.intercept_ - sum(original_coefficients * scaler.mean_)
```

This produced engine size = **+17.8581 g/km per litre**, MPG = **−5.0150 g/km per MPG**, and intercept = **329.1364 g/km**. The converted equation gives the same predictions as the standardized equation when applied consistently.

### `src/evaluator.py` — Measuring prediction errors

**What:** `evaluate_model(model, X_test, y_test)` generates predictions with `model.predict(X_test)` and calculates four regression metrics.

**Why:** Training a model is not enough: test-set metrics quantify prediction error and how much variation the model explains on unseen observations.

Let \(y_i\) be actual emissions, \(\hat y_i\) predicted emissions, \(\bar y\) the mean actual test emission, and n the number of test vehicles.

**Mean absolute error (MAE)**

\[
\mathrm{MAE}=\frac{1}{n}\sum_i|y_i-\hat y_i|
\]

The average absolute prediction error, in **g/km**. Here, **14.29 g/km**. Every absolute error contributes proportionally.

**Mean squared error (MSE)**

\[
\mathrm{MSE}=\frac{1}{n}\sum_i(y_i-\hat y_i)^2
\]

The average squared prediction error, in **(g/km)²**. Here, **466.11**. Large errors receive more weight because they are squared.

**Root mean squared error (RMSE)**

\[
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}
\]

Returns the squared-error measure to the target's original unit, **g/km**. Here, **21.59 g/km**. The implementation uses `np.sqrt(mse)` for compatibility across scikit-learn versions.

**Coefficient of determination (R²)**

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}
\]

Compares the model's squared error with the error from always predicting the mean of the actual test targets. Here, **0.8873**. A score of 1 is perfect; 0 matches that mean baseline; negative scores are possible when predictions perform worse than it.

### `src/visualizer.py` — Actual vs. predicted plot

**What:** `plot_predictions(y_test, predictions)` creates a 2D scatter plot with actual test emissions on the horizontal axis and predicted emissions on the vertical axis. It saves `outputs/actual_vs_predicted.png`.

**Why:** Summary metrics can hide patterns. The plot helps reveal systematic overprediction or underprediction, changing error spread, and outliers.

**How to read it:** Each dot represents a test vehicle. The dashed diagonal is **y = x**, meaning perfect prediction. Points above it are overpredictions; points below it are underpredictions. Vertical distance from the line is the prediction residual's magnitude. The dashed line is **not** the fitted regression line.

**Why no regression plane?** The fitted model has two predictors and one target, so its surface is a plane in three dimensions. This figure instead plots **actual versus predicted values**, which requires only two axes. A 3D plot was intentionally omitted from this project.

## Setup and execution

This project was developed in a shared Windows/PowerShell AI Engineering workspace. From the workspace root:

```powershell
# Create the environment once if it does not already exist
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install this project's dependencies
pip install -r .\02-Multiple-Linear-Regression\requirements.txt
```

Download the IBM CSV to:

```text
AI-Engineering-workspace/datasets/co2/FuelConsumptionCo2.csv
```

Run the project from the workspace root:

```powershell
python .\02-Multiple-Linear-Regression\main.py
```

Alternatively, from inside the project directory with the shared environment activated:

```powershell
python main.py
```

The entry point loads and preprocesses data, trains the model, prints its coefficients and evaluation metrics, and creates the actual-versus-predicted plot. To regenerate the correlation heatmap, call `analyze_correlation(df)` from `src.eda` after loading the DataFrame.

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` (see `requirements.txt`).

## What this project demonstrates

- Reading and exploring tabular data with pandas.
- Pearson correlation, heatmaps, and recognizing highly correlated features.
- Reproducible train/test splitting and prevention of preprocessing leakage.
- Standardization and the distinction between `fit`, `transform`, and `fit_transform`.
- Ordinary least squares with two predictors and interpretation of standardized versus original-unit coefficients.
- Evaluation using MAE, MSE, RMSE, and R².
- Interpreting an actual-versus-predicted scatter plot.

## Limitations

This is a learning exercise, not a production emissions estimator. The dataset represents 2014 vehicles, and results may not generalize to newer vehicle technologies or other markets. MPG and emissions are strongly related, but the relationship need not be exactly linear. Correlation and regression coefficients here are descriptive associations rather than proof of causality. The reported test metrics describe this particular random split, not guaranteed future performance.
