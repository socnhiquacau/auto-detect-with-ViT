#!/bin/bash
# Script to run Streamlit UI

echo "🚀 Starting Person Detection & Recognition UI..."
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit plotly pandas
fi

# Run Streamlit
echo "🌐 Starting Streamlit server..."
echo "Opening at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run streamlit_app.py
