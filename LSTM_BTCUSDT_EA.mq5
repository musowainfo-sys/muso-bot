//+------------------------------------------------------------------+
//|                                             LSTM_BTCUSDT_EA.mq5  |
//|                        LSTM-based Expert Advisor for Crypto/Forex|
//|                             Copyright 2026, Your Name          |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Your Name"
#property version   "1.00"
#property strict
#property description "LSTM-based EA for BTC/USD or other pairs using native MQL5 implementation"
#property indicator_chart_window
#property indicator_buffers 0

#include <Trade\Trade.mqh>
#include <Arrays\ArrayDouble.mqh>
#include "MyLSTM.mqh"

//--- Input parameters
input string          InpSymbol = "BTCUSDT";              // Trading symbol
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_H1;            // Data timeframe
input int             InpLookback = 20;                   // Lookback period for sequences
input int             InpHidden1 = 32;                    // First hidden layer size
input int             InpHidden2 = 16;                    // Second hidden layer size
input double          InpRiskPercent = 1.0;               // Risk percentage per trade
input double          InpConfidenceThreshold = 0.65;      // Minimum confidence for trading
input int             InpMaxBarsToFetch = 500;            // Max bars to fetch from API
input double          InpSLMultiplier = 1.5;              // Stop Loss multiplier (ATR)
input double          InpTPMultiplier = 3.0;              // Take Profit multiplier (ATR)
input int             InpMaxSpread = 100;                 // Maximum spread in points

//--- Global variables
CTrade                 m_trade;
CLSTM                  m_lstm;
CArrayDouble           m_prices_close;
CArrayDouble           m_prices_high;
CArrayDouble           m_prices_low;
CArrayDouble           m_prices_open;
CArrayDouble           m_volumes;
CArrayDouble           m_returns;
CArrayDouble           m_features_buffer;
CArrayDouble           m_predictions_log;
CArrayDouble           m_actual_returns_log;
CArrayDouble           m_confidences_log;

//--- Dashboard objects
string                 m_dashboard_label = "LSTM_EA_Dashboard";
double                 m_current_price = 0;
double                 m_last_prediction = 0;
double                 m_last_confidence = 0;
int                    m_total_trades = 0;
double                 m_total_pips = 0;
double                 m_sharpe_ratio = 0;
double                 m_accuracy = 0;

//--- Runtime variables
datetime               m_last_bar_time = 0;
bool                   m_model_loaded = false;
double                 m_atr_value = 0;
double                 m_avg_spread = 0;
double                 m_learning_rate = 0.01;
int                    m_feature_count = 5; // open, high, low, close, volume

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize arrays
   m_prices_close.Create(1000);
   m_prices_high.Create(1000);
   m_prices_low.Create(1000);
   m_prices_open.Create(1000);
   m_volumes.Create(1000);
   m_returns.Create(1000);
   m_features_buffer.Create(1000 * m_feature_count);
   m_predictions_log.Create(1000);
   m_actual_returns_log.Create(1000);
   m_confidences_log.Create(1000);
   
   // Set symbol for trading
   m_trade.SetExpertMagicNumber(123456);
   m_trade.SetDeviationInPoints(10);
   m_trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   // Create dashboard
   CreateDashboard();
   
   // Try to load existing model
   string model_file = "lstm_model_" + InpSymbol + ".bin";
   if(FileIsExist(model_file))
   {
      if(m_lstm.Load(model_file))
      {
         Print("LSTM model loaded successfully");
         m_model_loaded = true;
      }
      else
      {
         Print("Failed to load existing model, will initialize new one");
      }
   }
   
   // Initialize LSTM if not loaded
   if(!m_model_loaded)
   {
      int hidden_sizes[2] = {InpHidden1, InpHidden2};
      if(m_lstm.Initialize(m_feature_count, hidden_sizes, 2, 1))
      {
         Print("LSTM model initialized successfully");
         m_model_loaded = true;
      }
      else
      {
         Print("Failed to initialize LSTM model");
         return INIT_FAILED;
      }
   }
   
   // Calculate initial indicators
   CalculateATR();
   CalculateAvgSpread();
   
   // Initial data fetch and training
   if(!FetchAndProcessData())
   {
      Print("Warning: Failed to fetch initial data");
   }
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Save the model
   string model_file = "lstm_model_" + InpSymbol + ".bin";
   if(m_lstm.Save(model_file))
   {
      Print("Model saved successfully");
   }
   else
   {
      Print("Failed to save model");
   }
   
   // Clean up arrays
   m_prices_close.Destroy();
   m_prices_high.Destroy();
   m_prices_low.Destroy();
   m_prices_open.Destroy();
   m_volumes.Destroy();
   m_returns.Destroy();
   m_features_buffer.Destroy();
   m_predictions_log.Destroy();
   m_actual_returns_log.Destroy();
   m_confidences_log.Destroy();
   
   // Remove dashboard
   ObjectDelete(0, m_dashboard_label + "_bg");
   ObjectDelete(0, m_dashboard_label + "_title");
   ObjectDelete(0, m_dashboard_label + "_price");
   ObjectDelete(0, m_dashboard_label + "_pred");
   ObjectDelete(0, m_dashboard_label + "_conf");
   ObjectDelete(0, m_dashboard_label + "_perf");
   ObjectDelete(0, m_dashboard_label + "_trades");
   
   // Clean up LSTM
   // Note: The CLSTM destructor should handle cleanup automatically
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   static datetime last_processed_bar = 0;
   
   // Get current bar time
   datetime current_time = iTime(_Symbol, InpTimeframe, 0);
   
   // Process only if new bar appeared
   if(current_time != last_processed_bar)
   {
      last_processed_bar = current_time;
      
      // Update dashboard
      UpdateDashboard();
      
      // Fetch and process latest data
      if(FetchAndProcessData())
      {
         // Make prediction
         double prediction = 0;
         double confidence = 0;
         if(MakePrediction(prediction, confidence))
         {
            m_last_prediction = prediction;
            m_last_confidence = confidence;
            
            // Log prediction
            LogPrediction(m_current_price, prediction, confidence);
            
            // Check for trading signals
            if(MathAbs(confidence) > InpConfidenceThreshold)
            {
               ProcessTradingSignal(prediction, confidence);
            }
            
            // Update model with latest return
            UpdateModelOnline();
         }
      }
      
      // Update dashboard again with new info
      UpdateDashboard();
   }
}

//+------------------------------------------------------------------+
//| Create dashboard on chart                                        |
//+------------------------------------------------------------------+
void CreateDashboard()
{
   // Background rectangle
   ObjectCreate(0, m_dashboard_label + "_bg", OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_YDISTANCE, 10);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_XSIZE, 250);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_YSIZE, 160);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_BGCOLOR, clrBlack);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, m_dashboard_label + "_bg", OBJPROP_BACK, true);
   
   // Title
   ObjectCreate(0, m_dashboard_label + "_title", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, m_dashboard_label + "_title", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, m_dashboard_label + "_title", OBJPROP_TEXT, "LSTM EA Dashboard");
   ObjectSetInteger(0, m_dashboard_label + "_title", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, m_dashboard_label + "_title", OBJPROP_YDISTANCE, 15);
   ObjectSetInteger(0, m_dashboard_label + "_title", OBJPROP_COLOR, clrYellow);
   ObjectSetFont(0, m_dashboard_label + "_title", "Arial", 10, false, 0);
   
   // Price info
   ObjectCreate(0, m_dashboard_label + "_price", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, m_dashboard_label + "_price", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, m_dashboard_label + "_price", OBJPROP_TEXT, "Price: 0.00000");
   ObjectSetInteger(0, m_dashboard_label + "_price", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, m_dashboard_label + "_price", OBJPROP_YDISTANCE, 35);
   ObjectSetInteger(0, m_dashboard_label + "_price", OBJPROP_COLOR, clrWhite);
   ObjectSetFont(0, m_dashboard_label + "_price", "Arial", 9, false, 0);
   
   // Prediction info
   ObjectCreate(0, m_dashboard_label + "_pred", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, m_dashboard_label + "_pred", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, m_dashboard_label + "_pred", OBJPROP_TEXT, "Pred: 0.00000");
   ObjectSetInteger(0, m_dashboard_label + "_pred", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, m_dashboard_label + "_pred", OBJPROP_YDISTANCE, 50);
   ObjectSetInteger(0, m_dashboard_label + "_pred", OBJPROP_COLOR, clrWhite);
   ObjectSetFont(0, m_dashboard_label + "_pred", "Arial", 9, false, 0);
   
   // Confidence info
   ObjectCreate(0, m_dashboard_label + "_conf", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, m_dashboard_label + "_conf", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, m_dashboard_label + "_conf", OBJPROP_TEXT, "Conf: 0.000");
   ObjectSetInteger(0, m_dashboard_label + "_conf", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, m_dashboard_label + "_conf", OBJPROP_YDISTANCE, 65);
   ObjectSetInteger(0, m_dashboard_label + "_conf", OBJPROP_COLOR, clrWhite);
   ObjectSetFont(0, m_dashboard_label + "_conf", "Arial", 9, false, 0);
   
   // Performance info
   ObjectCreate(0, m_dashboard_label + "_perf", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, m_dashboard_label + "_perf", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, m_dashboard_label + "_perf", OBJPROP_TEXT, "Acc: 0%, SR: 0.00");
   ObjectSetInteger(0, m_dashboard_label + "_perf", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, m_dashboard_label + "_perf", OBJPROP_YDISTANCE, 80);
   ObjectSetInteger(0, m_dashboard_label + "_perf", OBJPROP_COLOR, clrWhite);
   ObjectSetFont(0, m_dashboard_label + "_perf", "Arial", 9, false, 0);
   
   // Trades info
   ObjectCreate(0, m_dashboard_label + "_trades", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, m_dashboard_label + "_trades", OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetString(0, m_dashboard_label + "_trades", OBJPROP_TEXT, "Trades: 0");
   ObjectSetInteger(0, m_dashboard_label + "_trades", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, m_dashboard_label + "_trades", OBJPROP_YDISTANCE, 95);
   ObjectSetInteger(0, m_dashboard_label + "_trades", OBJPROP_COLOR, clrWhite);
   ObjectSetFont(0, m_dashboard_label + "_trades", "Arial", 9, false, 0);
}

//+------------------------------------------------------------------+
//| Update dashboard information                                     |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   m_current_price = SymbolInfoDouble(InpSymbol, SYMBOL_BID);
   
   string price_text = StringFormat("Price: %.5f", m_current_price);
   ObjectSetString(0, m_dashboard_label + "_price", OBJPROP_TEXT, price_text);
   
   string pred_text = StringFormat("Pred: %.5f", m_last_prediction);
   ObjectSetString(0, m_dashboard_label + "_pred", OBJPROP_TEXT, pred_text);
   
   string conf_text = StringFormat("Conf: %.3f", m_last_confidence);
   ObjectSetString(0, m_dashboard_label + "_conf", OBJPROP_TEXT, conf_text);
   
   string perf_text = StringFormat("Acc: %.1f%%, SR: %.2f", m_accuracy*100, m_sharpe_ratio);
   ObjectSetString(0, m_dashboard_label + "_perf", OBJPROP_TEXT, perf_text);
   
   string trades_text = StringFormat("Trades: %d", m_total_trades);
   ObjectSetString(0, m_dashboard_label + "_trades", OBJPROP_TEXT, trades_text);
}

//+------------------------------------------------------------------+
//| Fetch market data from broker and API                            |
//+------------------------------------------------------------------+
bool FetchAndProcessData()
{
   // First, get data from the current chart
   int copied = CopyClose(InpSymbol, InpTimeframe, 0, InpMaxBarsToFetch, m_prices_close.ArrayPtr());
   if(copied < InpLookback + 10)  // Need extra bars for calculations
   {
      Print("Not enough data to process. Bars copied: ", copied);
      return false;
   }
   
   // Get other OHLC data
   CopyHigh(InpSymbol, InpTimeframe, 0, InpMaxBarsToFetch, m_prices_high.ArrayPtr());
   CopyLow(InpSymbol, InpTimeframe, 0, InpMaxBarsToFetch, m_prices_low.ArrayPtr());
   CopyOpen(InpSymbol, InpTimeframe, 0, InpMaxBarsToFetch, m_prices_open.ArrayPtr());
   CopyRealVolume(InpSymbol, InpTimeframe, 0, InpMaxBarsToFetch, m_volumes.ArrayPtr());
   
   // Calculate returns
   for(int i = 1; i < copied; i++)
   {
      double prev_close = m_prices_close.At(i-1);
      if(prev_close != 0)
         m_returns.Update(i, (m_prices_close.At(i) - prev_close) / prev_close);
      else
         m_returns.Update(i, 0);
   }
   
   // Update ATR and spread
   CalculateATR();
   CalculateAvgSpread();
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate ATR value                                              |
//+------------------------------------------------------------------+
void CalculateATR()
{
   double tr_array[];
   ArrayResize(tr_array, 14); // Using 14 periods for ATR calculation
   
   for(int i = 1; i < 14 && i < m_prices_high.GetSize(); i++)
   {
      double high_val = m_prices_high.At(i-1);
      double low_val = m_prices_low.At(i-1);
      double prev_close = m_prices_close.At(i);
      
      double tr = MathMax(high_val - low_val, MathMax(MathAbs(high_val - prev_close), MathAbs(low_val - prev_close)));
      tr_array[i-1] = tr;
   }
   
   double sum = 0;
   for(int i = 0; i < 13; i++)  // Skip first element which might be uninitialized
      sum += tr_array[i];
   
   m_atr_value = sum / 13.0;
   
   if(m_atr_value == 0 && m_prices_close.GetSize() > 0)
      m_atr_value = m_prices_close.At(0) * 0.001; // Fallback to 0.1% of current price
}

//+------------------------------------------------------------------+
//| Calculate average spread                                         |
//+------------------------------------------------------------------+
void CalculateAvgSpread()
{
   double spread_sum = 0;
   int count = MathMin(100, Bars(InpSymbol, InpTimeframe));
   
   for(int i = 0; i < count; i++)
   {
      double ask = iAsk(InpSymbol, InpTimeframe, i);
      double bid = iBid(InpSymbol, InpTimeframe, i);
      spread_sum += (ask - bid);
   }
   
   m_avg_spread = (count > 0) ? spread_sum / count / SymbolInfoDouble(InpSymbol, SYMBOL_POINT) : 0;
}

//+------------------------------------------------------------------+
//| Extract features from price data                                 |
//+------------------------------------------------------------------+
bool ExtractFeatures(double &features[][50], int &feature_count)
{
   int data_size = m_prices_close.GetSize();
   if(data_size < InpLookback + 1)
   {
      Print("Insufficient data for feature extraction. Required: ", InpLookback + 1, ", Available: ", data_size);
      return false;
   }
   
   feature_count = data_size - InpLookback;
   
   // Extract features for each sequence
   for(int i = 0; i < feature_count; i++)
   {
      int start_idx = i;
      int end_idx = i + InpLookback - 1;
      
      // For each timestep in the sequence
      for(int t = 0; t < InpLookback; t++)
      {
         int idx = start_idx + t;
         if(idx >= data_size) break;
         
         // Features: normalized open, high, low, close, volume
         double current_close = m_prices_close.At(idx);
         if(current_close == 0) continue; // Skip if invalid
         
         features[t][0] = (m_prices_open.At(idx) - current_close) / current_close;  // Open return
         features[t][1] = (m_prices_high.At(idx) - current_close) / current_close;  // High return
         features[t][2] = (m_prices_low.At(idx) - current_close) / current_close;   // Low return
         features[t][3] = (m_prices_close.At(idx) - current_close) / current_close; // Close return
         features[t][4] = MathLog(m_volumes.At(idx) + 1) / 10.0;                   // Log volume (normalized)
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Make prediction using the LSTM model                             |
//+------------------------------------------------------------------+
bool MakePrediction(double &prediction, double &confidence)
{
   if(m_prices_close.GetSize() < InpLookback)
   {
      Print("Insufficient data for prediction");
      return false;
   }
   
   // Prepare input sequence
   double input_seq[][];
   ArrayResize(input_seq, InpLookback);
   for(int i = 0; i < InpLookback; i++)
   {
      ArrayResize(input_seq[i], m_feature_count);
   }
   
   // Fill the sequence with the most recent data
   for(int t = 0; t < InpLookback; t++)
   {
      int idx = m_prices_close.GetSize() - InpLookback + t;
      if(idx < 0) continue;
      
      double current_close = m_prices_close.At(idx);
      if(current_close == 0) continue;
      
      input_seq[t][0] = (m_prices_open.At(idx) - current_close) / current_close;  // Open return
      input_seq[t][1] = (m_prices_high.At(idx) - current_close) / current_close;  // High return
      input_seq[t][2] = (m_prices_low.At(idx) - current_close) / current_close;   // Low return
      input_seq[t][3] = (m_prices_close.At(idx) - current_close) / current_close; // Close return
      input_seq[t][4] = MathLog(m_volumes.At(idx) + 1) / 10.0;                   // Log volume
   }
   
   // Make prediction
   double output[];
   if(m_lstm.Predict(input_seq, InpLookback, output))
   {
      prediction = output[0];
      
      // Calculate confidence as absolute value of prediction
      confidence = MathAbs(prediction);
      
      // Additional confidence measure based on how extreme the prediction is
      // compared to recent predictions
      if(m_predictions_log.GetSize() > 5)
      {
         double avg_recent_pred = 0;
         int count = MathMin(5, m_predictions_log.GetSize());
         for(int i = 0; i < count; i++)
         {
            avg_recent_pred += MathAbs(m_predictions_log.At(m_predictions_log.GetSize()-1-i));
         }
         avg_recent_pred /= count;
         
         if(avg_recent_pred > 0)
         {
            double relative_confidence = MathAbs(prediction) / avg_recent_pred;
            confidence = MathMin(confidence, relative_confidence);
         }
      }
      
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Process trading signal                                           |
//+------------------------------------------------------------------+
void ProcessTradingSignal(double prediction, double confidence)
{
   // Check if we already have an open position
   if(PositionSelect(InpSymbol))
   {
      // Already have a position, don't open another
      return;
   }
   
   // Determine trade direction based on prediction
   ENUM_ORDER_TYPE order_type = (prediction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   
   // Calculate lot size based on risk
   double lot_size = CalculateLotSize();
   if(lot_size <= 0)
   {
      Print("Invalid lot size calculated: ", lot_size);
      return;
   }
   
   // Calculate SL and TP
   double sl = 0, tp = 0;
   CalculateStopLossTakeProfit(order_type, sl, tp);
   
   // Place order
   if(order_type == ORDER_TYPE_BUY)
   {
      if(m_trade.Buy(lot_size, InpSymbol, 0, sl, tp, "LSTM_BUY"))
      {
         m_total_trades++;
         Print("BUY order placed: ", lot_size, " lots at ", SymbolInfoDouble(InpSymbol, SYMBOL_BID),
               ", SL: ", sl, ", TP: ", tp, ", Predicted move: ", prediction);
      }
   }
   else // SELL
   {
      if(m_trade.Sell(lot_size, InpSymbol, 0, sl, tp, "LSTM_SELL"))
      {
         m_total_trades++;
         Print("SELL order placed: ", lot_size, " lots at ", SymbolInfoDouble(InpSymbol, SYMBOL_ASK),
               ", SL: ", sl, ", TP: ", tp, ", Predicted move: ", prediction);
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                 |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (InpRiskPercent / 100.0);
   
   double tick_value = SymbolInfoDouble(InpSymbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_value <= 0) tick_value = 1; // Fallback value
   
   // Calculate stop loss distance in price terms
   double sl_distance = m_atr_value * InpSLMultiplier;
   if(sl_distance <= 0) sl_distance = SymbolInfoDouble(InpSymbol, SYMBOL_POINT) * 100; // Fallback
   
   // Calculate lot size
   double lot_size = risk_amount / sl_distance / tick_value;
   
   // Validate lot size
   double min_lot = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_STEP);
   
   lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
   
   // Round to lot step
   lot_size = MathRound(lot_size / lot_step) * lot_step;
   
   return lot_size;
}

//+------------------------------------------------------------------+
//| Calculate stop loss and take profit                              |
//+------------------------------------------------------------------+
void CalculateStopLossTakeProfit(ENUM_ORDER_TYPE order_type, double &sl, double &tp)
{
   double current_price = (order_type == ORDER_TYPE_BUY) ? 
                          SymbolInfoDouble(InpSymbol, SYMBOL_BID) : 
                          SymbolInfoDouble(InpSymbol, SYMBOL_ASK);
                          
   double sl_distance = m_atr_value * InpSLMultiplier;
   double tp_distance = m_atr_value * InpTPMultiplier;
   
   if(order_type == ORDER_TYPE_BUY)
   {
      sl = current_price - sl_distance;
      tp = current_price + tp_distance;
   }
   else // SELL
   {
      sl = current_price + sl_distance;
      tp = current_price - tp_distance;
   }
}

//+------------------------------------------------------------------+
//| Update model with latest return (online learning)               |
//+------------------------------------------------------------------+
void UpdateModelOnline()
{
   if(m_prices_close.GetSize() < 2)
      return;
   
   // Get the most recent return
   int latest_idx = m_prices_close.GetSize() - 1;
   int prev_idx = m_prices_close.GetSize() - 2;
   
   if(prev_idx < 0) return;
   
   double prev_close = m_prices_close.At(prev_idx);
   if(prev_close == 0) return;
   
   double actual_return = (m_prices_close.At(latest_idx) - prev_close) / prev_close;
   
   // Prepare input sequence for the bar that produced the return we're now observing
   double input_seq[][];
   ArrayResize(input_seq, InpLookback);
   for(int i = 0; i < InpLookback; i++)
   {
      ArrayResize(input_seq[i], m_feature_count);
   }
   
   // Fill the sequence with the data from the previous bar
   for(int t = 0; t < InpLookback; t++)
   {
      int idx = m_prices_close.GetSize() - 1 - InpLookback + t;  // -1 because we want data from before the return occurred
      if(idx < 0) continue;
      
      double current_close = m_prices_close.At(idx);
      if(current_close == 0) continue;
      
      input_seq[t][0] = (m_prices_open.At(idx) - current_close) / current_close;  // Open return
      input_seq[t][1] = (m_prices_high.At(idx) - current_close) / current_close;  // High return
      input_seq[t][2] = (m_prices_low.At(idx) - current_close) / current_close;   // Low return
      input_seq[t][3] = (m_prices_close.At(idx) - current_close) / current_close; // Close return
      input_seq[t][4] = MathLog(m_volumes.At(idx) + 1) / 10.0;                   // Log volume
   }
   
   // Adaptive learning rate based on market volatility
   double adaptive_lr = m_learning_rate;
   if(m_atr_value > 0)
   {
      // Increase learning rate during high volatility periods
      double volatility_factor = m_atr_value / (m_current_price * 0.001); // Normalize by price
      adaptive_lr *= MathMin(2.0, MathMax(0.5, volatility_factor));
   }
   
   // Update the model
   m_lstm.OnlineUpdate(input_seq, InpLookback, actual_return, adaptive_lr);
   
   // Log the update
   Print("Model updated with return: ", actual_return, ", LR: ", adaptive_lr);
}

//+------------------------------------------------------------------+
//| Log prediction and actual outcome                                |
//+------------------------------------------------------------------+
void LogPrediction(double current_price, double prediction, double confidence)
{
   // Add to logs
   m_predictions_log.Add(prediction);
   m_confidences_log.Add(confidence);
   
   // If we have previous data, log the actual return
   if(m_prices_close.GetSize() >= 2)
   {
      double prev_close = m_prices_close.At(m_prices_close.GetSize()-2);
      if(prev_close != 0)
      {
         double actual_return = (current_price - prev_close) / prev_close;
         m_actual_returns_log.Add(actual_return);
         
         // Calculate accuracy
         if(m_predictions_log.GetSize() > 1 && m_actual_returns_log.GetSize() > 1)
         {
            int correct_predictions = 0;
            double total_return = 0;
            double squared_returns_sum = 0;
            
            int start_idx = MathMax(0, m_predictions_log.GetSize() - 50); // Last 50 trades
            
            for(int i = start_idx; i < m_predictions_log.GetSize()-1; i++) // -1 because last prediction doesn't have actual yet
            {
               if((m_predictions_log.At(i) > 0 && m_actual_returns_log.At(i) > 0) ||
                  (m_predictions_log.At(i) < 0 && m_actual_returns_log.At(i) < 0))
                  correct_predictions++;
                  
               total_return += m_actual_returns_log.At(i);
               squared_returns_sum += m_actual_returns_log.At(i) * m_actual_returns_log.At(i);
            }
            
            int total_predictions = m_predictions_log.GetSize() - 1 - start_idx;
            if(total_predictions > 0)
            {
               m_accuracy = (double)correct_predictions / total_predictions;
               
               // Calculate Sharpe ratio (simplified)
               double avg_return = total_return / total_predictions;
               double variance = (squared_returns_sum / total_predictions) - (avg_return * avg_return);
               double std_dev = (variance > 0) ? MathSqrt(variance) : 0.0001; // Avoid division by zero
               
               if(std_dev > 0)
                  m_sharpe_ratio = avg_return / std_dev * MathSqrt(252); // Annualized
               else
                  m_sharpe_ratio = 0;
            }
         }
      }
   }
   
   // Keep logs at reasonable size
   if(m_predictions_log.GetSize() > 1000)
   {
      m_predictions_log.Shift(500, 0, 499);
      if(m_actual_returns_log.GetSize() > 1000)
         m_actual_returns_log.Shift(500, 0, 499);
      if(m_confidences_log.GetSize() > 1000)
         m_confidences_log.Shift(500, 0, 499);
   }
}

//+------------------------------------------------------------------+
//| WebRequest helper for fetching data from exchange APIs          |
//+------------------------------------------------------------------+
string FetchFromExchangeAPI(string endpoint)
{
   string api_url = "https://api.binance.com/api/v3/" + endpoint;
   
   string header = "";
   string post = "";
   string result = "";
   
   int res = WebRequest("GET", api_url, "", "User-Agent: MT5 Client", 5000, post, result, header);
   
   if(res == -1)
   {
      Print("WebRequest failed: ", GetLastError());
      return "";
   }
   
   return result;
}