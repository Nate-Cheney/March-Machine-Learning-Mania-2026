#!/bin/bash

source .venv/bin/activate

# Run Men's Scripts
echo "Running Men's scripts..."
python3 bin/m_clean_preprocess.py
python3 bin/m_train.py
python3 bin/m_clean_submission.py
python3 bin/m_test.py

# Run Women's Scripts
echo "Running Women's scripts..."
python3 bin/w_clean_preprocess.py 
python3 bin/w_train.py
python3 bin/w_clean_submission.py
python3 bin/w_test.py

# Create final submission
cat m_submission.csv > submission.csv
tail -n +2 w_submission.csv >> submission.csv

