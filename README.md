# ary_seq2seq

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

EN-ARY NMT school project

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Exploratory notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ary_seq2seq and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ary_seq2seq   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ary_seq2seq a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── ary_kh.py           <- Keras-Hub implementation of the model          
    │   └── {colmo,layers}.py   <- Support code for custom layers
    │
    └── dataset.py              <- Scripts to download or generate data
```

--------

Tested w/ Python 3.12 (as `tensorflow-text` is not packaged for anything higher at the time of writing).

## Setup the environment

The environment expects to be managed via [uv](https://docs.astral.sh/uv/getting-started/installation/).


```bash
make create_environment
source .venv/bin/activate
make requirements
```

NOTE: Downloading the dataset requires to be logged in to HF, and to have accepted the T&C for the `atlasia/Atlaset`.

```bash
python ary_seq2seq/dataset.py
```

Training the Hub variant can be started via
```bash
python ary_seq2seq/modeling/ary_kh.py [--with-swiglu]
```
