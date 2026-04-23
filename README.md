# HSES AES-CPA Project

Project made by Irene Cerván, Pablo Esteban, Andreea Kapás, Oriol López Petit, Xiao Li Segarra.

## Installation

First craete a virtual environment and use it with: 

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install required dependencies with:

```bash
python -m pip install -r requirements.txt
```

Now you should be able to execute the code.

## Project structure

The files of the code are the following:

- `cpa_aes/data.py`: Code for dataset loading.
- `cpa_aes/cpa.py`: CPA core and key recovery logic.
- `cpa_aes/cli.py`: Command-line interface.

## Run on dataset1

In our case we placed the contents of dataset1.zip in a directory called dataset1 and used the following command in order to 
retrieve the key of the first exercise:

```bash
python -m cpa_aes.cli --dataset-dir dataset1 --output-csv outputs/dataset1_cpa_results.csv
```

## Run on dataset2

In our case we placed the contents of dataset2.zip in a directory called dataset2 and used the following command in order to 
retrieve the key of the second exercise:

```bash
python -m cpa_aes.cli --dataset-dir dataset2 --output-csv outputs/dataset2_cpa_results.csv --clock
```
