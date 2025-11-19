#!/bin/bash

# If anything fails, stop immediately — but show a nice error and KEEP the terminal open
set -euo pipefail

# Catch any error and jump to our handler
trap 'handle_error $? $LINENO' ERR

handle_error() {
    local exit_code=$1
    local line_number=$2
    echo ""
    echo "=================================================================="
    echo "ERROR: Setup failed on line $line_number with exit code $exit_code"
    echo "=================================================================="
    echo "Most common fixes:"
    echo "   • Make sure you have Python 3.9+ installed (run: python3 --version)"
    echo " • Try deleting the old venv first: rm -rf venv"
    echo " • On some systems you need: python3 -m pip install --upgrade pip"
    echo ""
    echo "The terminal will stay open so you can read this message."
    echo "Close it when you're ready."
    # shellcheck disable=SC2086
    exec bash  # this keeps the window open (Linux/macOS/WSL)
    # On pure macOS you can also use: read -rp "Press Enter to continue..."
}

# ——— Your actual setup starts here ———

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "=================================================================="
echo "Setup completed successfully! 🎉"
echo "To activate later: source venv/bin/activate"
echo "To run: python h1-dyadic-law.py --n_dyads 500 --seed 42"
echo "=================================================================="

# Only reach this point if setup succeeded → now pause so the user can read
echo ""
read -rp "Press Enter to close this window..." 
echo "Goodbye!"
