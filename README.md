# Data science course project

## General notes

* Teacher's website: <http://web.ist.utl.pt/~claudia.antunes/DSLabs/>
* Health dataset in Kaggle: <https://www.kaggle.com/datasets/brandao/diabetes>
* Climate dataset in Kaggle: <https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data>
* Notes: <https://docs.google.com/document/d/1EKG_WaoMWat4D9Xw-C6WE-Ayf509Ai2a45vYr1PP4r4/edit>

## Setup and installation

1. Download above datasets into the corresponding resources folder

2. Install all the required packages, run the following command:

```bash
pip install -r requirements.txt
```

3. Extract health dataset, run the following command:

```bash
unzip health/resources/health.zip -d health/resources/data
```

4. Extract climate dataset, run the following command:

```bash
unzip climate/resources/climate.zip -d climate/resources/data && unzip climate/resources/drought.csv.zip -d climate/resources/data
```
