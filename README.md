# TCS Stock Trading Project — README

Welcome — this repository contains a Jupyter Notebook `TCS_Trad.ipynb` that demonstrates a simple workflow to download historical market data, build a target label (next-day up/down), join prior-day predictors, and set up a Random Forest classifier. This README explains the notebook step-by-step, how to run it interactively, and provides clear, copy‑pasteable code snippets to reproduce and extend the work.

Notebook (source)
- TCS_Trad.ipynb: https://github.com/Niramaynextgen/tcsstocktradproject/blob/f51834907895794d888fbf24c0fcc67d07d5a5c9/TCS_Trad.ipynb

Table of contents
- Project summary
- Requirements
- Installation
- Quick start (run the notebook)
- Step-by-step explanation of the notebook
- Reproducible code snippets (prepare, train, evaluate)
- Interactive usage (ipywidgets)
- Suggestions & next steps
- Notes and warnings
- Contributing & License

Project summary
This project:
- Downloads historical index data (example used: `^NSEI`) with yfinance.
- Caches the data to a JSON file so repeated runs are faster.
- Builds a binary target indicating whether the next close is higher than the previous close.
- Builds predictors by shifting prior-day feature values into today's row.
- Sets up a RandomForestClassifier as a baseline.
The notebook displays early exploratory tables and plots, and prepares the dataset for modeling.

Requirements
- Python 3.9+ (or 3.8)
- Jupyter / JupyterLab
- Packages:
  - pandas
  - yfinance
  - scikit-learn
  - matplotlib
  - ipywidgets (optional, for interactivity)
Install using pip:

pip install pandas yfinance scikit-learn matplotlib jupyterlab ipywidgets

Installation
1. Clone this repository:
   git clone https://github.com/Niramaynextgen/tcsstocktradproject.git
2. Change directory:
   cd tcsstocktradproject
3. (Recommended) Create a virtual environment and activate it.
4. Install the packages above.

Quick start — run the notebook locally
1. Start Jupyter:
   jupyter lab
   or
   jupyter notebook
2. Open `TCS_Trad.ipynb` and run cells top to bottom.

How the notebook works (cell-by-cell summary)
- Cell 1 — Imports and data fetching.
  - Uses `yfinance.Ticker('<TICKER>')` to download history via `.history(period='max')`.
  - Stores the data to a JSON file (filename `tcs_ata.json` in the notebook) if the file doesn't exist; otherwise loads cached JSON to avoid repeated downloads.
  - Displays `.head()` of the historical OHLCV DataFrame.
- Cell 2 — Plot:
  - Plots `Close` price history using pandas `plot.line`.
- Cell 3 — Build target:
  - Creates a `data` dataframe with column `Actual_Close` (rename of `Close`).
  - Builds `Target` using a rolling-2 apply: whether the current close is greater than the prior close (binary 1/0).
- Cell 4 — Create prior-row predictors:
  - Creates `nsfi_pr = nsfi_hist.copy().shift(1)` so predictors represent previous day's OHLCV values.
- Cell 5 — Join predictors to `data`:
  - Joins predictors `['Close','Volume','Open','High','Low']` into `data` and drops the initial NaN row.
  - Now each row contains today's `Actual_Close` and `Target`, and the prior day's features (predictors).
- Cell 6 — Set up model:
  - Imports `RandomForestClassifier` and creates `model = RandomForestClassifier(n_estimators=100, max_depth=200, random_state=1)`.
- Cell 7 — (Notebook currently displays scikit-learn visual repr). The notebook doesn't currently show training, evaluation, or backtest steps — the README includes code to continue.

Reproducible code snippets
Below are complete, copy-pasteable Python functions you can put into a script or a notebook cell to reproduce and extend the notebook.

1) Data preparation function
```python
import os
import pandas as pd
import yfinance as yf

def fetch_and_cache(ticker='^NSEI', cache_path='tcs_ata.json', period='max'):
    """
    Fetches historical data using yfinance and caches to JSON.
    Returns a DataFrame indexed by datetime with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits.
    """
    if os.path.exists(cache_path):
        df = pd.read_json(cache_path)
        # ensure index is datetime
        df.index = pd.to_datetime(df.index)
        return df
    else:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        df.to_json(cache_path)
        return df
```

2) Prepare modeling DataFrame (target + prior-day predictors)
```python
def prepare_model_data(hist_df):
    """
    hist_df: historical OHLCV DataFrame (indexed by date)
    Returns: DataFrame with columns: Actual_Close, Target, Close, Volume, Open, High, Low
    where predictors are prior day's values (shifted).
    """
    data = hist_df[['Close']].rename(columns={'Close':'Actual_Close'})
    # Target: whether today > previous day (binary). Use rolling(2) apply to Close series.
    data['Target'] = hist_df['Close'].rolling(2).apply(lambda x: 1.0 if x.iloc[1] > x.iloc[0] else 0.0)
    # Make prior-day predictors
    prior = hist_df.copy().shift(1)
    predictors = ['Close','Volume','Open','High','Low']
    data = data.join(prior[predictors]).iloc[1:]  # drop first row which will contain NaNs
    return data.dropna()
```

3) Train / evaluate baseline Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_and_evaluate(df):
    # df expected from prepare_model_data(); Target is binary {0.0,1.0}
    X = df[['Close','Volume','Open','High','Low']]
    y = df['Target'].astype(int)

    # Time-series aware split: use earliest N% for training, last (1-N%) for testing to avoid leakage.
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(n_estimators=100, max_depth=200, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("Test accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_test, preds))
    return model, X_test, y_test, preds
```

4) Example minimal end-to-end usage:
```python
hist = fetch_and_cache('^NSEI', cache_path='tcs_ata.json')
df = prepare_model_data(hist)
model, X_test, y_test, preds = train_and_evaluate(df)
```

Interactive usage (ipywidgets)
Make the notebook interactive so you can choose ticker, date range or change model hyperparameters on the fly. Below is a minimal example using ipywidgets:

```python
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

ticker_widget = widgets.Text(value='^NSEI', description='Ticker:')
reload_button = widgets.Button(description='Load & Run')

def on_reload(b):
    ticker = ticker_widget.value.strip()
    hist = fetch_and_cache(ticker, cache_path=f'{ticker}_cache.json')
    df = prepare_model_data(hist)
    model, X_test, y_test, preds = train_and_evaluate(df)
    # Simple plot of predictions vs actual
    plt.figure(figsize=(12,4))
    plt.plot(y_test.values, label='Actual (1=up,0=down)', alpha=0.7)
    plt.plot(preds, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title(f'Prediction vs Actual for {ticker}')
    plt.show()

reload_button.on_click(on_reload)
display(widgets.HBox([ticker_widget, reload_button]))
```

Binder / Running in the cloud (optional)
You can make the notebook runnable interactively on Binder:
- Add a `requirements.txt` with the package list and create a binder badge:
  - Example `requirements.txt`:
    pandas
    yfinance
    scikit-learn
    matplotlib
    ipywidgets
- Create a binder badge (manually) using https://mybinder.org with the GitHub repo URL and path to the notebook.
Note: offline caching or large data downloads may affect start-up time.

Suggestions & next steps (improvements)
- Use a time-series split / walk-forward cross-validation (avoid random shuffles).
- Add more features:
  - Returns (log returns), moving averages, RSI, MACD, day-of-week, volatility measures.
- Normalize or scale features where appropriate (though tree-based models are scale-invariant).
- Hyperparameter tune RandomForest (GridSearchCV with time-series aware CV).
- Evaluate economic performance: simulate a simple trading strategy and measure return, drawdown, Sharpe ratio.
- Persist trained model with joblib or pickle and add a simple prediction API.
- Add robust error handling for yfinance download failures and missing data.
- Add better caching with Parquet for performance.
- Label caution: the `Target` in this notebook is a simplistic next-day up/down indicator — not a guaranteed profitable signal.

Notes and warnings
- This project is educational. Do not use the model for real trading without rigorous validation and risk controls.
- Financial data is non-stationary. A model that worked historically may fail in live markets.
- Avoid data leakage: ensure predictors come only from information available at prediction time (the notebook uses prior-day predictors which is correct, but additional engineering must preserve this property).

Contributing
- Contributions and improvements are welcome. Open issues or pull requests with a clear description of the change.
- Add tests for the data preparation functions and a requirements file for reproducibility.

License
- Add a license file to the repository if you want to make your intentions explicit (MIT, Apache 2.0, etc).

If you'd like, I can:
- Create a `requirements.txt` file for you.
- Improve the notebook by adding the training and evaluation cells and an interactive widget cell.
- Generate a downloadable script (e.g., `train.py`) that runs the whole pipeline end-to-end.

Which of those would you like me to do next?
