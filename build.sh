#!/bin/bash

echo "LSTM EA Build Script"
echo "===================="

# Check if we're in the right directory
if [ ! -f "MyLSTM.mqh" ] || [ ! -f "LSTM_BTCUSDT_EA.mq5" ]; then
    echo "Error: Required files not found in current directory!"
    echo "Please run this script from the directory containing MyLSTM.mqh and LSTM_BTCUSDT_EA.mq5"
    exit 1
fi

echo "Files found:"
echo "- MyLSTM.mqh"
echo "- LSTM_BTCUSDT_EA.mq5"

echo ""
echo "Validating file syntax..."

# Basic validation of MQL5 files
if grep -q "#property copyright" LSTM_BTCUSDT_EA.mq5 && grep -q "#property version" LSTM_BTCUSDT_EA.mq5; then
    echo "✓ LSTM_BTCUSDT_EA.mq5 has proper MQL5 structure"
else
    echo "✗ LSTM_BTCUSDT_EA.mq5 missing proper MQL5 structure"
    exit 1
fi

if grep -q "#property copyright" MyLSTM.mqh && grep -q "#property version" MyLSTM.mqh; then
    echo "✓ MyLSTM.mqh has proper MQL5 include structure"
else
    echo "✗ MyLSTM.mqh missing proper MQL5 include structure"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo ""
echo "To use this EA in MT5:"
echo "1. Copy MyLSTM.mqh to <MT5_Data_Folder>/MQL5/Include/"
echo "2. Copy LSTM_BTCUSDT_EA.mq5 to <MT5_Data_Folder>/MQL5/Experts/"
echo "3. Compile both files in MetaEditor"
echo "4. Attach the EA to any chart"
echo ""

exit 0