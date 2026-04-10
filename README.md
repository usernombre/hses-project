# HSES AES-CPA Project

## Project structure

- `cpa_aes/data.py`: dataset loading
- `cpa_aes/cpa.py`: CPA core and key recovery logic
- `cpa_aes/cli.py`: command-line interface

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run on dataset1

```bash
python -m cpa_aes.cli --dataset-dir dataset1 --output-csv outputs/dataset1_cpa_results.csv
```