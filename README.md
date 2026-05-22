# MA Pricescraper

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Heat Pump Pricescraper is a data analysis project focused on collecting, processing, and analyzing heat pump market pricing and product characteristics. The repository provides tools for web scraping, manufacturer matching, data modeling, and includes a demo app for interactive price transparency using Streamlit.


## Streamlit price transparency demo

The repository includes a lightweight Streamlit app in `app.py`. It lets users enter a heat pump offer price and key technical characteristics used by the price model. The app predicts an expected product/configuration price with the `RandomForest_200t` model setup from the analysis notebook and applies a broad +/-20% reasonableness band.

Run locally:

```powershell
.\.venv\python.exe -m streamlit run app.py
```

For free public hosting, there are two options:

- Streamlit Community Cloud with `streamlit_app/streamlit_app.py` as the main file. This runs the live Python/Streamlit version.
- GitHub Pages from the `docs/` folder. This is a static version that uses a precomputed RandomForest prediction grid and a client-side interactive table.

To refresh the GitHub Pages data bundle after changing the model or dataset:

```powershell
.\.venv\python.exe scripts\build_github_pages_data.py
```

See `DEPLOYMENT.md` for hosting notes.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── Pricescraper   <- Source code. Execute the scripts in the order of their numbering.
    │
    ├── 2_scrapers            <- Webscraper for the 14 selected websites.
    │
    ├── 4_matching               <- Matching scripts for 13 selected manufacturers.
    │
    ├── 6_analysis              <- Data analysis scripts.
    │
    ├── 1_filter_hplib.py             <- Python script to filter the original hplib.
    │
    ├── 3_combine_and_standardize_scraped_data.py             <- Python file to organize the  │                                                            scraped data.
    │
    └── 5_combine_check_decoding_matched_manufacturers.py             <- Python script to   generate the final database for analysis.
```

--------

