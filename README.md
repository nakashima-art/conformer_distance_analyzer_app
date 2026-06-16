# Conformer Distance Analyzer

A Streamlit app for evaluating NOE-compatible H···H distances in conformer ensembles from multi-conformer SDF files or Gaussian log files.

## Features
- Read multi-conformer SDF files
- Read multiple Gaussian log files
- Group conformers by candidate stereochemistry
- Define atom mappings and interchangeable proton sets
- Evaluate H···H distance criteria
- Compare summed conformer populations

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
