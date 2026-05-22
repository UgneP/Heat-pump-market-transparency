from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


APP_TITLE = "Heat Pump Price Transparency"
DATA_PATH = Path(__file__).parent / "data" / "processed" / "filtered_hplib_w_prices_and_decoding.csv"


NUMERIC_COLUMNS = [
    "Price",
    "SCOP",
    "Rated Power low T [kW]",
    "Rated Power medium T [kW]",
    "Storage Size (L)",
    "eta low T [%]",
    "eta medium T [%]",
    "SPL outdoor high Power [dBA]",
    "Max. water heating temperature [\N{DEGREE SIGN}C]",
    "Poff [W]",
    "PSB [W]",
]


MODEL_NUMERIC_FEATURES = [
    "Rated Power medium T [kW]",
    "Storage Size (L)",
    "SPL outdoor high Power [dBA]",
    "Poff [W]",
]
MODEL_CATEGORICAL_FEATURES = ["Config", "Storage", "Refrigerant"]
MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
EXPECTED_PRICE_BAND = 0.20


FEATURE_LABELS = {
    "Rated Power medium T [kW]": "Rated power",
    "Storage Size (L)": "Storage size",
    "SPL outdoor high Power [dBA]": "Noise",
    "Poff [W]": "Standby power",
    "Config": "Configuration",
    "Storage": "Tank configuration",
    "Refrigerant": "Refrigerant type",
}


MANUFACTURER_SHORT_NAMES = {
    "bosch thermotechnik gmbh": "Bosch",
    "bosch thermotechnik gmbh (buderus)": "Buderus",
    "viessmann climate solutions se": "Viessmann",
    "ait-deutschland gmbh": "AIT",
    "daikin europe n.v.": "Daikin",
    "samsung electronics air conditioner europe b.v.": "Samsung",
    "wolf gmbh": "Wolf",
    "mitsubishi electric air conditioning systems europe ltd": "Mitsubishi",
    "panasonic marketing europe gmbh": "Panasonic",
    "lg electronics inc.": "LG",
    "toshiba air conditioning": "Toshiba",
    "vaillant gmbh": "Vaillant",
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":house_with_garden:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #17211b;
                --muted: #5f6f67;
                --line: #dce5df;
                --surface: #f7faf8;
                --green: #26734d;
                --yellow: #c88a13;
                --orange: #c95f20;
                --red: #a33b35;
            }
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            h1, h2, h3 {
                letter-spacing: 0;
                color: var(--ink);
            }
            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.75rem 0.9rem;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 0.75rem;
                margin: 0.75rem 0 1rem 0;
            }
            .metric-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 0.85rem 0.95rem;
                min-width: 0;
            }
            .metric-label {
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.2;
                margin-bottom: 0.35rem;
            }
            .metric-value {
                color: var(--ink);
                font-size: clamp(1.35rem, 2.2vw, 2rem);
                font-weight: 650;
                line-height: 1.05;
                overflow-wrap: anywhere;
            }
            .score-panel {
                border: 1px solid var(--line);
                border-radius: 8px;
                padding: 1rem;
                background: #ffffff;
            }
            .label-pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.35rem 0.7rem;
                color: white;
                font-weight: 700;
                margin-bottom: 0.7rem;
            }
            .muted {
                color: var(--muted);
                font-size: 0.94rem;
            }
            .small-note {
                color: var(--muted);
                font-size: 0.84rem;
            }
            .range-value {
                color: var(--ink);
                font-size: clamp(1.8rem, 3vw, 2.8rem);
                font-weight: 750;
                line-height: 1.05;
                margin: 0.2rem 0 1rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(value: object) -> str:
    return clean_text(value).lower()


def short_manufacturer(name: object) -> str:
    key = normalize_text(name)
    if "johnson" in key and "hitachi" in key:
        return "Johnson / Hitachi"
    return MANUFACTURER_SHORT_NAMES.get(key, clean_text(name))


def format_eur(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"EUR {value:,.0f}".replace(",", "'")


def format_number(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def metric_grid(items: list[tuple[str, str]]) -> None:
    cards = "\n".join(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>"
        for label, value in items
    )
    st.markdown(f"<div class='metric-grid'>{cards}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [col.replace("\u00c2", "").strip() for col in df.columns]

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    df = df[df["Price"].notna()].copy()
    df = df[df["Price"] > 0].copy()
    df["Manufacturer display"] = df["Manufacturer"].apply(short_manufacturer)
    df["Configuration"] = (
        df["Config"].fillna("unknown").astype(str).str.strip()
        + " - "
        + df["Storage"].fillna("unknown").astype(str).str.strip()
    )
    df["Price per kW"] = df["Price"] / df["Rated Power low T [kW]"].replace(0, np.nan)
    df["Offer name"] = df["Manufacturer display"] + " | " + df["Titel"].fillna(df["Model"]).astype(str)
    df["Power band"] = pd.cut(
        df["Rated Power low T [kW]"],
        bins=[0, 6, 9, 12, np.inf],
        labels=["up to 6 kW", "6-9 kW", "9-12 kW", "over 12 kW"],
        include_lowest=True,
    )
    return df


def option_list(series: pd.Series) -> list[str]:
    values = [clean_text(value) for value in series.dropna().unique()]
    return sorted([value for value in values if value])


def model_training_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["Price"] + MODEL_FEATURES).copy()


@st.cache_data(show_spinner=False)
def build_offer_presets(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    model_df = model_training_data(df)
    model_df = model_df.copy()
    model_df["Model power band"] = pd.cut(
        model_df["Rated Power medium T [kW]"],
        bins=[0, 6, 9, 12, 16, np.inf],
        labels=["up to 6 kW", "6-9 kW", "9-12 kW", "12-16 kW", "over 16 kW"],
        include_lowest=True,
    )
    grouped = (
        model_df.groupby(["Config", "Storage", "Refrigerant", "Model power band"], dropna=False, observed=True)
        .agg(
            observations=("Price", "count"),
            price=("Price", "median"),
            rated_power=("Rated Power medium T [kW]", "median"),
            storage_size=("Storage Size (L)", "median"),
            noise=("SPL outdoor high Power [dBA]", "median"),
            standby=("Poff [W]", "median"),
        )
        .reset_index()
        .sort_values(["observations", "price"], ascending=[False, True])
    )

    presets = {"Custom offer": {}}
    for _, row in grouped.head(18).iterrows():
        label = (
            f"{clean_text(row['Config']).title()} / {clean_text(row['Storage'])} / "
            f"{clean_text(row['Refrigerant'])} / {clean_text(row['Model power band'])}"
        )
        presets[label] = {
            "Config": clean_text(row["Config"]),
            "Storage": clean_text(row["Storage"]),
            "Refrigerant": clean_text(row["Refrigerant"]),
            "Rated Power medium T [kW]": float(row["rated_power"]),
            "Storage Size (L)": float(row["storage_size"]),
            "SPL outdoor high Power [dBA]": float(row["noise"]),
            "Poff [W]": float(row["standby"]),
            "Price": float(row["price"]),
        }
    return presets


@st.cache_resource(show_spinner=False)
def train_price_model(df: pd.DataFrame) -> dict[str, object]:
    model_df = model_training_data(df)
    X = model_df[MODEL_FEATURES]
    y = model_df["Price"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), MODEL_NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                MODEL_CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(random_state=0, n_estimators=200, max_depth=10),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    percentage_errors = np.abs((y_test - y_pred) / y_test)

    return {
        "model": model,
        "training_rows": len(model_df),
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mape": float(percentage_errors.mean() * 100),
        "feature_importance": grouped_feature_importance(model),
    }


def grouped_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    transformed_names = preprocessor.get_feature_names_out()

    rows = []
    for raw_name, importance in zip(transformed_names, regressor.feature_importances_):
        name = raw_name.split("__", 1)[-1]
        feature = next(
            (column for column in MODEL_FEATURES if name == column or name.startswith(f"{column}_")),
            name,
        )
        rows.append({"feature": FEATURE_LABELS.get(feature, feature), "importance": float(importance)})

    grouped = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    grouped["importance"] = grouped["importance"] / grouped["importance"].sum()
    return grouped


def expected_price_label(price: float, low: float, high: float, expected_price: float | None = None) -> dict[str, object]:
    ratio = price / expected_price if expected_price else np.nan
    if price < low:
        label, color = "Cheaper than expected - check details", "#6f7772"
        message = "Congratulations on a cheaper price, but check whether everything is included and whether the manufacturer or supplier is reliable."
    elif price <= high:
        label, color = "In expected range", "#26734d"
        message = "This offer is within the model-based expected range for the entered characteristics."
    else:
        label, color = "Higher than expected", "#a33b35"
        message = "This offer is above the expected range; check whether extra services, warranty, installation scope, or availability explain the premium."

    return {"label": label, "color": color, "low": low, "high": high, "ratio": ratio, "message": message}


def prediction_range(model: Pipeline, training_data: pd.DataFrame, user_inputs: dict[str, object | None]) -> tuple[float, float, float, list[str]]:
    scenarios = training_data[MODEL_FEATURES].copy()
    missing_fields = []

    for feature, value in user_inputs.items():
        if value is None or value == "Not specified":
            missing_fields.append(FEATURE_LABELS.get(feature, feature))
        else:
            scenarios[feature] = value

    scenarios = scenarios.drop_duplicates().reset_index(drop=True)
    predictions = model.predict(scenarios)
    central_price = float(np.median(predictions))
    low = float(np.min(predictions) * (1 - EXPECTED_PRICE_BAND))
    high = float(np.max(predictions) * (1 + EXPECTED_PRICE_BAND))
    return low, high, central_price, missing_fields


def score_message(score: dict[str, object]) -> str:
    if "message" in score:
        return str(score["message"])
    label = str(score.get("label", ""))
    if "Cheaper" in label:
        return "Congratulations on a cheaper price, but check whether everything is included and whether the manufacturer or supplier is reliable."
    if "Higher" in label:
        return "This offer is above the expected range; check whether extra services, warranty, installation scope, or availability explain the premium."
    return "This offer is within the model-based expected range for the entered characteristics."


def render_expected_price_result(score: dict[str, object], user_price: float, missing_fields: list[str]) -> None:
    note = score_message(score)
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        note = f"{note} Range is wider because these inputs were not specified: {missing_text}."
    st.markdown(
        f"""
        <div class='score-panel'>
            <span class='label-pill' style='background:{score['color']}'>{score['label']}</span>
            <div class='metric-label'>Expected price range</div>
            <div class='range-value'>{format_eur(score['low'])} - {format_eur(score['high'])}</div>
            <p class='small-note'>Your offer: {format_eur(user_price)}</p>
            <p class='small-note'>{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_offer_checker(df: pd.DataFrame) -> None:
    st.subheader("Air-to-Water Heat Pump Price Check")
    st.markdown(
        "<p class='muted'>Estimate whether an offer price is below, within, or above the model-based expected range.</p>",
        unsafe_allow_html=True,
    )

    model_info = train_price_model(df)
    model = model_info["model"]
    defaults = model_training_data(df)
    numeric_limits = {
        feature: (float(defaults[feature].min()), float(defaults[feature].max()))
        for feature in MODEL_NUMERIC_FEATURES
    }

    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.caption("Only the offer price is required. Leave technical fields blank if they are unknown.")
        with st.form("offer_form"):
            form_cols = st.columns(2)
            with form_cols[0]:
                price = st.number_input(
                    "Offer price",
                    min_value=100.0,
                    max_value=100000.0,
                    value=12000.0,
                    step=250.0,
                )
                power_kw = st.number_input(
                    "Rated power (kW)",
                    min_value=round(numeric_limits["Rated Power medium T [kW]"][0], 1),
                    max_value=round(numeric_limits["Rated Power medium T [kW]"][1], 1),
                    value=None,
                    step=0.1,
                    placeholder=f"{numeric_limits['Rated Power medium T [kW]'][0]:.1f}-{numeric_limits['Rated Power medium T [kW]'][1]:.1f}",
                    help="Limited to the observed range in the research dataset.",
                )
                storage_size = st.number_input(
                    "Storage size (L)",
                    min_value=round(numeric_limits["Storage Size (L)"][0], 0),
                    max_value=round(numeric_limits["Storage Size (L)"][1], 0),
                    value=None,
                    step=10.0,
                    placeholder=f"{numeric_limits['Storage Size (L)'][0]:.0f}-{numeric_limits['Storage Size (L)'][1]:.0f}",
                    help="Limited to the observed range in the research dataset.",
                )
            with form_cols[1]:
                config_options = ["Not specified"] + option_list(defaults["Config"])
                storage_options = ["Not specified"] + option_list(defaults["Storage"])
                refrigerant_options = ["Not specified"] + option_list(defaults["Refrigerant"])
                config = st.selectbox(
                    "Configuration",
                    config_options,
                    index=0,
                )
                storage = st.selectbox(
                    "Tank configuration",
                    storage_options,
                    index=0,
                )
                refrigerant = st.selectbox(
                    "Refrigerant type",
                    refrigerant_options,
                    index=0,
                )
            advanced_cols = st.columns(2)
            with advanced_cols[0]:
                sound = st.number_input(
                    "Noise (dBA)",
                    min_value=round(numeric_limits["SPL outdoor high Power [dBA]"][0], 1),
                    max_value=round(numeric_limits["SPL outdoor high Power [dBA]"][1], 1),
                    value=None,
                    step=0.5,
                    placeholder=f"{numeric_limits['SPL outdoor high Power [dBA]'][0]:.1f}-{numeric_limits['SPL outdoor high Power [dBA]'][1]:.1f}",
                    help="Limited to the observed range in the research dataset.",
                )
            with advanced_cols[1]:
                standby = st.number_input(
                    "Standby power (W)",
                    min_value=round(numeric_limits["Poff [W]"][0], 1),
                    max_value=round(numeric_limits["Poff [W]"][1], 1),
                    value=None,
                    step=0.5,
                    placeholder=f"{numeric_limits['Poff [W]'][0]:.1f}-{numeric_limits['Poff [W]'][1]:.1f}",
                    help="Limited to the observed range in the research dataset.",
                )
            submitted = st.form_submit_button("Check price", use_container_width=True)

    low, high, expected_price, missing_fields = prediction_range(
        model,
        defaults,
        {
            "Rated Power medium T [kW]": power_kw,
            "Storage Size (L)": storage_size,
            "SPL outdoor high Power [dBA]": sound,
            "Poff [W]": standby,
            "Config": config,
            "Storage": storage,
            "Refrigerant": refrigerant,
        },
    )
    score = expected_price_label(price, low, high, expected_price)

    with right:
        render_expected_price_result(score, price, missing_fields)

def render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Market Dashboard")
    st.markdown(
        "<p class='muted'>Explore simple benchmark tables and matched offers from the research dataset.</p>",
        unsafe_allow_html=True,
    )

    display_columns = [
        "Manufacturer display",
        "Titel",
        "Price",
        "Rated Power low T [kW]",
        "SCOP",
        "Configuration",
        "Refrigerant",
        "Website",
        "Date_scraped",
    ]
    column_labels = {
        "Manufacturer display": "Manufacturer",
        "Titel": "Product title",
        "Rated Power low T [kW]": "Power kW",
    }

    filtered = df.copy()
    with st.expander("Table filters", expanded=True):
        query = st.text_input("Search manufacturer, model, title, or website", placeholder="e.g. Vaillant, R290, idealo")
        filter_cols = st.columns(3)
        with filter_cols[0]:
            manufacturers = st.multiselect(
                "Manufacturer",
                option_list(df["Manufacturer display"]),
                placeholder="All manufacturers",
            )
            price_range = st.slider(
                "Price",
                min_value=int(df["Price"].min()),
                max_value=int(df["Price"].max()),
                value=(int(df["Price"].min()), int(df["Price"].max())),
                step=250,
            )
        with filter_cols[1]:
            configurations = st.multiselect("Configuration", option_list(df["Configuration"]), placeholder="All configurations")
            power_range = st.slider(
                "Rated power kW",
                min_value=float(df["Rated Power low T [kW]"].min()),
                max_value=float(df["Rated Power low T [kW]"].max()),
                value=(float(df["Rated Power low T [kW]"].min()), float(df["Rated Power low T [kW]"].max())),
                step=0.5,
            )
        with filter_cols[2]:
            refrigerants = st.multiselect("Refrigerant type", option_list(df["Refrigerant"]), placeholder="All refrigerants")
            scop_range = st.slider(
                "SCOP",
                min_value=float(df["SCOP"].min()),
                max_value=float(df["SCOP"].max()),
                value=(float(df["SCOP"].min()), float(df["SCOP"].max())),
                step=0.05,
            )

    if query:
        needle = query.lower()
        filtered = filtered[
            filtered[["Manufacturer display", "Model", "Titel", "Model/Type", "Website", "Refrigerant"]]
            .fillna("")
            .astype(str)
            .apply(lambda row: row.str.lower().str.contains(needle, regex=False).any(), axis=1)
        ]
    if manufacturers:
        filtered = filtered[filtered["Manufacturer display"].isin(manufacturers)]
    if configurations:
        filtered = filtered[filtered["Configuration"].isin(configurations)]
    if refrigerants:
        filtered = filtered[filtered["Refrigerant"].isin(refrigerants)]
    filtered = filtered[
        filtered["Price"].between(*price_range)
        & filtered["Rated Power low T [kW]"].between(*power_range)
        & filtered["SCOP"].between(*scop_range)
    ]

    if filtered.empty:
        st.warning("No offers match the current table filters.")
        return

    metric_grid(
        [
            ("Offers", f"{len(filtered):,}"),
            ("Manufacturers", str(filtered["Manufacturer display"].nunique())),
            ("Median price", format_eur(filtered["Price"].median())),
            ("Median SCOP", format_number(filtered["SCOP"].median(), 2)),
            ("Median EUR/kW", format_eur(filtered["Price per kW"].median())),
        ]
    )

    brand_summary = (
        filtered.groupby("Manufacturer display", as_index=False)
        .agg(
            offers=("Price", "count"),
            median_price=("Price", "median"),
            median_price_per_kw=("Price per kW", "median"),
            median_scop=("SCOP", "median"),
        )
        .sort_values("median_price", ascending=False)
    )
    config_summary = (
        filtered.groupby("Configuration", as_index=False)
        .agg(offers=("Price", "count"), median_price=("Price", "median"), median_scop=("SCOP", "median"))
        .sort_values("median_price", ascending=False)
    )

    table_cols = st.columns(2, gap="large")
    with table_cols[0]:
        st.markdown("**Manufacturer benchmarks**")
        st.dataframe(
            brand_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "median_price": st.column_config.NumberColumn("Median price", format="EUR %.0f"),
                "median_price_per_kw": st.column_config.NumberColumn("Median EUR/kW", format="EUR %.0f"),
                "median_scop": st.column_config.NumberColumn("Median SCOP", format="%.2f"),
            },
        )
    with table_cols[1]:
        st.markdown("**Configuration benchmarks**")
        st.dataframe(
            config_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "median_price": st.column_config.NumberColumn("Median price", format="EUR %.0f"),
                "median_scop": st.column_config.NumberColumn("Median SCOP", format="%.2f"),
            },
        )

    st.markdown("**Matched offers**")
    st.dataframe(
        filtered.sort_values("Price")[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Manufacturer display": "Manufacturer",
            "Titel": "Product title",
            "Price": st.column_config.NumberColumn("Price", format="EUR %.0f"),
            "Rated Power low T [kW]": st.column_config.NumberColumn("Power kW", format="%.1f"),
            "SCOP": st.column_config.NumberColumn("SCOP", format="%.2f"),
        },
    )


def render_methodology(df: pd.DataFrame) -> None:
    model_info = train_price_model(df)

    st.subheader("Methodology")
    st.markdown(
        f"""
        This demo is based on the market-transparency study *Increasing market transparency of residential
        air-to-water heat pumps in Europe by clustering the appliances* by Ugne Potthoff, Felix Morlock,
        and Felix Wortmann, prepared for HPC2026.

        **Data basis**

        The underlying research links certified technical data from the Heat Pump Library / Keymark records
        with web-scraped European online retail prices. The processed app dataset contains **{len(df):,}
        matched air-to-water heat pump observations** from 13 manufacturers. The paper focuses on residential
        AWHPs up to 13 kW using R32 or R290 refrigerants. Product names from retailer listings were matched
        to Keymark entries through a reference dictionary because naming is inconsistent across vendors,
        countries, and manufacturer catalogues.

        **What the price checker estimates**

        The checker predicts an expected **equipment/product configuration price**, not a full installed system
        cost. It excludes installation, accessories, subsidies, delivery conditions, warranty differences,
        installer margin, and site-specific work. The app model uses rated power, storage size, refrigerant type,
        noise, standby power, configuration, and tank configuration.
        """
    )

    st.markdown("**Model Error And Interpretation**")
    metric_grid(
        [
            ("Holdout R2", f"{model_info['r2']:.2f}"),
            ("Mean absolute error", format_eur(model_info["mae"])),
            ("Mean percentage error", f"{model_info['mape']:.1f}%"),
        ]
    )
    st.markdown(
        f"""
        The app follows the notebook's `RandomForest_200t` specification:
        `RandomForestRegressor(random_state=0, n_estimators=200, max_depth=10)`.
        The metrics above are calculated from the same holdout split for the model used in this app, so the
        reported R2, mean absolute error, and mean percentage error describe one consistent predictive fit.
        On the current processed dataset, the holdout error is about **{format_eur(model_info["mae"])}** on
        average, or **{model_info["mape"]:.1f}%** of the observed price on average. An individual prediction
        can still be off by roughly this amount, and sometimes more. For this reason the app uses a deliberately
        broad **+/-20% expected range** rather than a narrow point estimate.

        The paper reports that rated power, storage size, refrigerant type, noise, standby power, reference COP,
        and tank configuration help explain much of the observed price variation. SCOP and maximum water
        temperature showed only marginal effects in that analysis.

        **Important: predictive fit is not causality.** The model estimates associations learned from observed
        online product prices. It does not prove that changing one feature causes the price to change by a
        specific amount. Features are correlated in the market: for example, refrigerant type, monoblock/split
        configuration, manufacturer origin, storage tanks, and product positioning often appear together.
        Therefore the result should be read as a benchmarking signal, not a causal explanation or purchasing
        recommendation.
        """
    )

    st.markdown(
        """
        **How the label is assigned**

        - `Cheaper than expected`: offer price is more than 20% below the model prediction
        - `In expected range`: offer price is within +/-20% of the model prediction
        - `Higher than expected`: offer price is more than 20% above the model prediction

        A cheaper-than-expected result is not automatically a better offer. It may indicate a good deal, but it
        can also signal missing components, a narrower scope, less support, unclear warranty terms, or a less
        reliable supplier. A higher-than-expected result can likewise be justified if it includes additional
        services or scope not represented in the dataset.

        **Reference**

        Potthoff, U., Morlock, F., & Wortmann, F. *Increasing market transparency of residential air-to-water
        heat pumps in Europe by clustering the appliances*. HPC2026 working paper.
        """
    )
    st.info(
        "This is a research demo. It benchmarks product/configuration offer prices only. It should not be used as a final purchasing recommendation."
    )
    st.download_button(
        "Download processed benchmark dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="heat_pump_price_benchmark_dataset.csv",
        mime="text/csv",
    )


def main() -> None:
    inject_styles()
    df = load_data()

    st.title(APP_TITLE)
    st.caption("A research demo for checking heat pump offer prices against matched European online market data.")

    tab_checker, tab_dashboard, tab_methodology = st.tabs(["Price check", "Dashboard", "Methodology"])
    with tab_checker:
        render_offer_checker(df)
    with tab_dashboard:
        render_dashboard(df)
    with tab_methodology:
        render_methodology(df)


if __name__ == "__main__":
    main()
