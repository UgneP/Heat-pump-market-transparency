# Demo app deployment

This repository includes a small Streamlit demo for heat pump price transparency:

```powershell
.\.venv\python.exe -m streamlit run app.py
```

The app lets a user enter a heat pump offer price plus key technical attributes, then benchmarks the offer against matched European online market observations from `data/processed/filtered_hplib_w_prices_and_decoding.csv`.

## Option 1: Streamlit Community Cloud

Use Streamlit Community Cloud for the most professional free deployment:

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io.
3. Create a new app from the GitHub repository.
4. Set the main file path to `streamlit_app/streamlit_app.py`.
5. Streamlit Community Cloud will use `streamlit_app/requirements.txt` because it is in the same directory as the app entrypoint.

This is the most faithful version because it runs the Python model pipeline directly.

## Option 2: GitHub Pages

GitHub Pages cannot run Streamlit or Python, so this repository also includes a static version in `docs/`.
It uses:

- `docs/index.html`, `docs/styles.css`, and `docs/app.js` for the interface
- `docs/data/app-data.json` for the public data bundle
- a precomputed RandomForest prediction grid generated from the app model
- a client-side interactive table for the matched offers

To refresh the static data after changing the model or dataset:

```powershell
.\.venv\python.exe scripts\build_github_pages_data.py
```

To enable GitHub Pages:

1. Push the repository to GitHub.
2. Open the repository settings on GitHub.
3. Go to **Pages**.
4. Set **Source** to **Deploy from a branch**.
5. Select the `main` branch and the `/docs` folder.
6. Save. GitHub will publish the site after the Pages build finishes.

The static version is an approximation of the live model behavior. It does not execute the Random Forest in the browser; it filters a precomputed prediction grid.

## Other free options

Hugging Face Spaces can also host Streamlit apps for free. Choose the Streamlit SDK and use the same package pins from `streamlit_app/requirements.txt`.

Render and Railway can host Python web apps, but their free tiers change more often and may sleep aggressively. For this project, Streamlit Community Cloud is the cleanest path.
