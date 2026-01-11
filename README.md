# SANIMA Stock Price Time Series Analysis

A comprehensive time series analysis and forecasting notebook for SANIMA stock market data using Python, ARIMA modeling, and technical indicators.

## Overview

This project provides an end-to-end time series analysis of SANIMA stock prices, including exploratory data analysis, statistical modeling, forecasting, and technical indicator calculations. The notebook is designed to run seamlessly in Google Colab.

## Features

### 1. Data Exploration & Visualization
- Interactive candlestick charts
- Price trends with moving averages (7-day, 30-day, 90-day)
- Volume analysis
- Daily returns distribution
- Correlation heatmaps
- Yearly performance statistics

### 2. Time Series Analysis
- **Seasonal Decomposition**: Breaks down the series into trend, seasonality, and residual components
- **Stationarity Testing**: Augmented Dickey-Fuller (ADF) test
- **ACF/PACF Analysis**: Identifies autocorrelation patterns for model selection

### 3. ARIMA Modeling
- **Auto ARIMA**: Automatically finds optimal model parameters (p, d, q)
- **Train-Test Split**: 80-20 split for model validation
- **Performance Metrics**: RMSE, MAE, MAPE
- **Residual Analysis**: Ensures model assumptions are met

### 4. Forecasting
- Future price predictions with confidence intervals
- 30-day ahead forecasting
- Interactive visualization of historical data and forecasts

### 5. Technical Indicators
- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Trend momentum indicator
- **Bollinger Bands**: Volatility and price range analysis

## Dataset

The analysis uses SANIMA stock data with the following columns:
- `published_date`: Trading date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `per_change`: Percentage change
- `traded_quantity`: Volume traded
- `traded_amount`: Total amount traded
- `status`: Trading status

## Requirements

### Python Libraries
```
pandas
numpy
matplotlib
seaborn
plotly
statsmodels
pmdarima
scikit-learn
scipy
```

All dependencies are automatically installed when you run the first cell of the notebook in Google Colab.

## Getting Started

### Option 1: Google Colab (Recommended)

1. **Upload the Notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Upload `sanima_time_series_analysis.ipynb`

2. **Upload Your Data**
   - Run the data loading cell
   - When prompted, upload your `SANIMA.csv` file
   - Alternatively, mount Google Drive and load from there

3. **Run the Analysis**
   - Execute cells sequentially from top to bottom
   - The first cell installs required packages (takes ~1-2 minutes)
   - Review outputs and visualizations as you progress

### Option 2: Local Jupyter Notebook

1. **Install Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn plotly statsmodels pmdarima scikit-learn scipy jupyter
   ```

2. **Launch Jupyter**
   ```bash
   jupyter notebook sanima_time_series_analysis.ipynb
   ```

3. **Update Data Path**
   - Modify the data loading cell to point to your local CSV file path
   - Comment out the Google Colab file upload code

## Usage Guide

### Running the Complete Analysis

Execute all cells in order:

```python
# 1. Install packages (Colab only)
!pip install statsmodels pmdarima plotly -q

# 2. Load data
# Upload SANIMA.csv when prompted

# 3. Run all subsequent cells
# Each section builds on the previous one
```

### Customizing the Analysis

**Change the forecast horizon:**
```python
# In section 10, modify:
future_steps = 60  # Forecast 60 days instead of 30
```

**Adjust ARIMA parameters manually:**
```python
# In section 9, specify custom order:
model = ARIMA(train_data, order=(5, 1, 2))  # (p, d, q)
```

**Modify train-test split:**
```python
# In section 7, change split ratio:
train_size = int(len(df) * 0.9)  # Use 90% for training
```

**Change moving average windows:**
```python
# In section 2, customize periods:
df['MA_14'] = df['close'].rolling(window=14).mean()
df['MA_50'] = df['close'].rolling(window=50).mean()
```

## Output Interpretation

### Model Performance Metrics

- **RMSE (Root Mean Squared Error)**: Lower is better. Measures average prediction error in the same units as the target variable.
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.
- **MAPE (Mean Absolute Percentage Error)**: Percentage error, easier to interpret across different price levels.

### Stationarity Test Results

- **p-value < 0.05**: Series is stationary (good for ARIMA)
- **p-value > 0.05**: Series is non-stationary (may need differencing)

### Technical Indicators

**RSI Values:**
- RSI > 70: Overbought (potential sell signal)
- RSI < 30: Oversold (potential buy signal)
- RSI 30-70: Neutral

**MACD:**
- MACD crosses above Signal Line: Bullish signal
- MACD crosses below Signal Line: Bearish signal

**Bollinger Bands:**
- Price touches upper band: Potentially overbought
- Price touches lower band: Potentially oversold
- Price within bands: Normal trading range

## Project Structure

```
.
├── sanima_time_series_analysis.ipynb    # Main analysis notebook
├── SANIMA.csv                            # Stock data (to be uploaded)
└── README.md                             # This file
```

## Notebook Sections

1. **Setup and Data Loading**: Import libraries and load dataset
2. **Data Preprocessing**: Clean, transform, and prepare data
3. **Exploratory Data Analysis**: Visualizations and statistical summaries
4. **Time Series Decomposition**: Trend, seasonality, and residual analysis
5. **Stationarity Testing**: ADF test for time series properties
6. **ACF and PACF Analysis**: Autocorrelation patterns
7. **Train-Test Split**: Prepare data for modeling
8. **ARIMA Model - Auto Selection**: Find optimal parameters
9. **ARIMA Model Training and Forecasting**: Build and evaluate model
10. **Future Forecasting**: Predict future prices
11. **Technical Indicators**: Calculate and visualize trading signals
12. **Summary and Conclusions**: Overall analysis results

## Tips for Best Results

1. **Data Quality**: Ensure your CSV file has no missing dates or corrupted values
2. **Sufficient Data**: More historical data (1+ years) improves model accuracy
3. **Parameter Tuning**: The Auto ARIMA may take several minutes to find optimal parameters
4. **Interpreting Forecasts**: Remember that longer-term forecasts have higher uncertainty
5. **Market Context**: Combine technical analysis with fundamental analysis for trading decisions

## Troubleshooting

### Common Issues

**Issue**: Package installation fails
```bash
# Solution: Install packages individually
!pip install statsmodels
!pip install pmdarima
!pip install plotly
```

**Issue**: File upload not working in Colab
```python
# Solution: Use Google Drive
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/SANIMA.csv')
```

**Issue**: Auto ARIMA takes too long
```python
# Solution: Reduce parameter search space
auto_model = auto_arima(
    train_data,
    max_p=3, max_q=3,  # Reduce from 5
    stepwise=True
)
```

**Issue**: Memory error with large datasets
```python
# Solution: Analyze recent data only
df = df.tail(1000)  # Use last 1000 days
```

## Limitations

- Stock prices are influenced by many external factors not captured in historical price data alone
- ARIMA models assume linear patterns and may not capture complex non-linear relationships
- Past performance does not guarantee future results
- This analysis is for educational purposes and should not be considered financial advice

## Future Enhancements

Potential improvements to consider:

- [ ] Add LSTM/GRU neural network models
- [ ] Implement GARCH for volatility modeling
- [ ] Include sentiment analysis from news data
- [ ] Add Prophet for handling multiple seasonality
- [ ] Implement walk-forward validation
- [ ] Add portfolio optimization features
- [ ] Include comparison with market indices

## Contributing

Feel free to fork this project and customize it for your needs. Suggestions for improvements are welcome!

## Disclaimer

This notebook is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always conduct thorough research and consult with financial advisors before making investment decisions.

## 👤 Author

**SabinAdhikari**  
GitHub: [SabinAdhikarii](https://github.com/SabinAdhikarii)

---

## 📞 Support

For questions or issues, please contact me at sabinofficial99@gmail.com).

---

**Last Updated:** January 2026

## License

This project is for educational and demonstration purposes. The Sanima bank dataset has been used. You can use the dataset directltly. The original source of the dataset is scrapped data from [NEPSE](https://nepsealpha.com/stocks/SANIMA/info?utm_source=copilot.com)  



**Happy Analyzing! 📈**
