# LSTM-based Expert Advisor for MetaTrader 5

## Overview

This project implements a fully native LSTM-based Expert Advisor (EA) for MetaTrader 5 (MT5) that performs algorithmic trading on cryptocurrency and forex markets. The EA uses a custom-built LSTM neural network entirely implemented in MQL5 without any external dependencies.

## Key Features

### 1. Native MQL5 LSTM Implementation
- Complete LSTM neural network built from scratch in MQL5
- No external libraries or dependencies
- Configurable architecture (input, hidden layers, output)
- Online learning capabilities

### 2. Real-time Data Processing
- Fetches live market data from broker
- Feature engineering with returns, high-low spreads, volume
- Min-max normalization for stable training

### 3. Adaptive Trading System
- Automated buy/sell decisions based on LSTM predictions
- Dynamic position sizing based on risk parameters
- ATR-based stop-loss and take-profit levels
- Confidence threshold filtering

### 4. Performance Monitoring
- Real-time dashboard displaying key metrics
- Accuracy and Sharpe ratio tracking
- Trade history logging
- Model persistence

### 5. Online Learning
- Continuous model updates with new market data
- Adaptive learning rates based on market volatility
- Regime detection for changing market conditions

## Architecture

### MyLSTM.mqh
The core LSTM implementation includes:
- `CLSTMCell` class: Single LSTM unit with forget, input, output, and candidate gates
- `CLSTM` class: Multi-layer LSTM network with configurable architecture
- Forward propagation methods
- Online learning and weight updates
- Model serialization (save/load functionality)

### LSTM_BTCUSDT_EA.mq5
The main Expert Advisor with:
- Data acquisition and preprocessing
- Feature extraction pipeline
- Trading logic and execution
- Dashboard display
- Performance tracking

## Input Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| InpSymbol | BTCUSDT | Trading symbol |
| InpTimeframe | PERIOD_H1 | Data timeframe |
| InpLookback | 20 | Sequence length for LSTM |
| InpHidden1 | 32 | Size of first hidden layer |
| InpHidden2 | 16 | Size of second hidden layer |
| InpRiskPercent | 1.0 | Risk percentage per trade |
| InpConfidenceThreshold | 0.65 | Minimum prediction confidence |
| InpMaxBarsToFetch | 500 | Max bars to fetch for analysis |
| InpSLMultiplier | 1.5 | Stop-loss multiplier (ATR) |
| InpTPMultiplier | 3.0 | Take-profit multiplier (ATR) |

## How It Works

### 1. Data Collection
The EA collects OHLCV data from the broker for the specified symbol and timeframe.

### 2. Feature Engineering
- Calculates returns from closing prices
- Normalizes features using min-max scaling
- Creates sequences of specified lookback length

### 3. Prediction
- Feeds the latest sequence into the LSTM model
- Generates a prediction for the next return
- Calculates confidence score

### 4. Trading Decision
- If confidence exceeds threshold, evaluates trade direction
- Calculates position size based on risk parameters
- Sets stop-loss and take-profit levels based on ATR
- Executes trade if no position is currently open

### 5. Online Learning
- After each bar closes, calculates the actual return
- Updates the LSTM model with the new data point
- Adjusts learning rate based on market volatility

## Dashboard Display

The EA provides a real-time dashboard showing:
- Current market price
- Latest LSTM prediction
- Prediction confidence
- Trading accuracy
- Sharpe ratio
- Total trades executed

## Installation

1. Copy both files to your MT5 Data folder:
   - `MQL5/Experts/LSTM_BTCUSDT_EA.mq5`
   - `MQL5/Include/MyLSTM.mqh`

2. Compile the files in MetaEditor

3. Attach the EA to any chart with the desired symbol

4. Configure input parameters as needed

## Risk Management

- Position sizing based on account balance and risk percentage
- ATR-based stop-loss and take-profit levels
- Single position limit per symbol
- Confidence threshold prevents low-quality trades

## Backtesting and Live Trading

The EA supports both backtesting and live trading modes. For backtesting, ensure sufficient historical data is available for the selected symbol and timeframe.

## Notes

- This is an advanced implementation combining neural networks with algorithmic trading
- Past performance does not guarantee future results
- Always test thoroughly on demo accounts before live trading
- Consider market conditions and adjust parameters accordingly

## License

Copyright 2026. This code is provided as-is without warranty. Use at your own risk.