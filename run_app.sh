#!/bin/bash
cd "$(dirname "$0")"
streamlit run app.py --theme.base light --theme.primaryColor "#A100FF" --theme.secondaryBackgroundColor "#F7F1FF"
