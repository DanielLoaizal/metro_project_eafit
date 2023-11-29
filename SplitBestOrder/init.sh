#!/bin/bash
pip install -r $(dirname "$0")/requirements.txt
python3 $(dirname "$0")/main.py
