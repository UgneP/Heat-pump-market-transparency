from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


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
]


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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
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


def filter_market(
    df: pd.DataFrame,
    manufacturer: str,
    config: str,
    refrigerant: str,
    power_kw: float,
    scop: float | None,
    strictness: str,
) -> tuple[pd.DataFrame, list[str]]:
    pool = df.copy()
    filters = []

    if config != "Any":
        pool = pool[pool["Configuration"] == config]
        filters.append(f"same package type ({config})")

    if refrigerant != "Any":
        ref_pool = pool[pool["Refrigerant"].astype(str) == refrigerant]
        if len(ref_pool) >= 8:
            pool = ref_pool
            filters.append(f"same refrigerant ({refrigerant})")

    if manufacturer != "Any":
        brand_pool = pool[pool["Manufacturer display"] == manufacturer]
        if len(brand_pool) >= 8:
            pool = brand_pool
            filters.append(f"same manufacturer ({manufacturer})")

    power_window = {"Tight": 1.5, "Balanced": 2.5, "Broad": 4.0}[strictness]
    power_pool = pool[
        pool["Rated Power low T [kW]"].between(power_kw - power_window, power_kw + power_window)
    ]
    if len(power_pool) >= 8:
        pool = power_pool
        filters.append(f"similar rated power (+/-{power_window:g} kW)")

    if scop is not None and not pd.isna(scop):
        scop_window = {"Tight": 0.25, "Balanced": 0.45, "Broad": 0.75}[strictness]
        scop_pool = pool[pool["SCOP"].between(scop - scop_window, scop + scop_window)]
        if len(scop_pool) >= 8:
            pool = scop_pool
            filters.append(f"similar SCOP (+/-{scop_window:g})")

    if len(pool) < 8:
        pool = df[df["Rated Power low T [kW]"].between(power_kw - 4.0, power_kw + 4.0)]
        filters = ["fallback: similar rated power across the dataset"]

    if len(pool) < 8:
        pool = df.copy()
        filters = ["fallback: full matched dataset"]

    return pool, filters


def score_offer(price: float, market: pd.DataFrame) -> dict[str, object]:
    prices = market["Price"].dropna()
    q10, q25, q50, q75, q90 = prices.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    percentile = float((prices <= price).mean() * 100)
    ratio_to_median = float(price / q50) if q50 else np.nan

    if percentile <= 25:
        label, color = "Good price", "#26734d"
    elif percentile <= 60:
        label, color = "Fair price", "#c88a13"
    elif percentile <= 85:
        label, color = "Expensive", "#c95f20"
    else:
        label, color = "Very expensive", "#a33b35"

    return {
        "label": label,
        "color": color,
        "percentile": percentile,
        "ratio_to_median": ratio_to_median,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q90": float(q90),
    }


def nearest_comparables(df: pd.DataFrame, price: float, power_kw: float, scop: float | None) -> pd.DataFrame:
    comparable = df.copy()
    comparable["distance"] = (
        ((comparable["Price"] - price) / comparable["Price"].std(ddof=0)).abs().fillna(0)
        + ((comparable["Rated Power low T [kW]"] - power_kw) / comparable["Rated Power low T [kW]"].std(ddof=0)).abs().fillna(0)
    )
    if scop is not None and not pd.isna(scop) and comparable["SCOP"].notna().sum() > 3:
        comparable["distance"] += (
            ((comparable["SCOP"] - scop) / comparable["SCOP"].std(ddof=0)).abs().fillna(0)
        )
    columns = [
        "Manufacturer display",
        "Titel",
        "Price",
        "Rated Power low T [kW]",
        "SCOP",
        "Configuration",
        "Refrigerant",
        "Website",
    ]
    return comparable.sort_values("distance").head(8)[columns]


def draw_price_gauge(score: dict[str, object], user_price: float) -> None:
    q10 = score["q10"]
    q90 = score["q90"]
    span_min = min(q10, user_price) * 0.92
    span_max = max(q90, user_price) * 1.08
    gauge = pd.DataFrame(
        {
            "Marker": ["10th percentile", "25th", "Median", "75th", "90th", "Your offer"],
            "Price": [score["q10"], score["q25"], score["median"], score["q75"], score["q90"], user_price],
            "y": [1, 1, 1, 1, 1, 1.08],
        }
    )
    st.scatter_chart(gauge, x="Price", y="y", color="Marker", height=180)
    st.caption(f"Benchmark range shown from {format_eur(span_min)} to {format_eur(span_max)}.")


def render_offer_checker(df: pd.DataFrame) -> None:
    st.subheader("Offer Checker")
    st.markdown(
        "<p class='muted'>Compare a heat pump offer with matched online market observations from the research dataset.</p>",
        unsafe_allow_html=True,
    )

    defaults = df.dropna(subset=["Rated Power low T [kW]", "SCOP"]).copy()
    median_power = float(defaults["Rated Power low T [kW]"].median())
    median_scop = float(defaults["SCOP"].median())

    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        with st.form("offer_form"):
            form_cols = st.columns(2)
            with form_cols[0]:
                price = st.number_input("Offer price", min_value=100.0, max_value=100000.0, value=12000.0, step=250.0)
                power_kw = st.number_input(
                    "Rated power at low temperature (kW)",
                    min_value=1.0,
                    max_value=40.0,
                    value=round(median_power, 1),
                    step=0.1,
                )
                scop = st.number_input("SCOP", min_value=1.0, max_value=8.0, value=round(median_scop, 2), step=0.05)
            with form_cols[1]:
                manufacturer = st.selectbox("Manufacturer", ["Any"] + option_list(df["Manufacturer display"]))
                config = st.selectbox("Package type", ["Any"] + option_list(df["Configuration"]))
                refrigerant = st.selectbox("Refrigerant", ["Any"] + option_list(df["Refrigerant"]))
            strictness = st.segmented_control(
                "Comparison scope",
                ["Tight", "Balanced", "Broad"],
                default="Balanced",
                help="Tighter comparisons use narrower power and SCOP windows when enough comparable offers exist.",
            )
            submitted = st.form_submit_button("Check offer", use_container_width=True)

    market, filters = filter_market(df, manufacturer, config, refrigerant, power_kw, scop, strictness)
    score = score_offer(price, market)

    with right:
        st.markdown("<div class='score-panel'>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='label-pill' style='background:{score['color']}'>{score['label']}</span>",
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(3)
        metric_cols[0].metric("Compared with", f"{len(market)} offers")
        metric_cols[1].metric("Price percentile", f"{score['percentile']:.0f}%")
        metric_cols[2].metric("Median benchmark", format_eur(score["median"]))

        below_above = "above" if score["ratio_to_median"] >= 1 else "below"
        st.write(
            f"Your offer is **{abs(score['ratio_to_median'] - 1) * 100:.0f}% {below_above}** "
            f"the median of the selected comparison group."
        )
        st.write(
            f"A typical central range for comparable offers is **{format_eur(score['q25'])} to {format_eur(score['q75'])}**."
        )
        st.markdown(
            "<p class='small-note'>This label benchmarks product/package prices in the dataset. It does not include installation, subsidies, availability, warranty differences, or site-specific work.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("Comparison basis: " + ", ".join(filters) + ".")
    draw_price_gauge(score, price)

    st.markdown("**Closest comparable observations**")
    comparables = nearest_comparables(market, price, power_kw, scop).rename(
        columns={
            "Manufacturer display": "Manufacturer",
            "Titel": "Product title",
            "Rated Power low T [kW]": "Power kW",
        }
    )
    st.dataframe(
        comparables,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="EUR %.0f"),
            "Power kW": st.column_config.NumberColumn("Power kW", format="%.1f"),
            "SCOP": st.column_config.NumberColumn("SCOP", format="%.2f"),
        },
    )


def filtered_dashboard_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Dashboard filters")
    manufacturers = st.sidebar.multiselect(
        "Manufacturers",
        option_list(df["Manufacturer display"]),
        default=[],
        placeholder="All manufacturers",
    )
    configs = st.sidebar.multiselect(
        "Package types",
        option_list(df["Configuration"]),
        default=[],
        placeholder="All package types",
    )
    refrigerants = st.sidebar.multiselect(
        "Refrigerants",
        option_list(df["Refrigerant"]),
        default=[],
        placeholder="All refrigerants",
    )
    price_min, price_max = int(df["Price"].min()), int(df["Price"].max())
    price_range = st.sidebar.slider(
        "Price range",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=250,
    )

    filtered = df[df["Price"].between(*price_range)].copy()
    if manufacturers:
        filtered = filtered[filtered["Manufacturer display"].isin(manufacturers)]
    if configs:
        filtered = filtered[filtered["Configuration"].isin(configs)]
    if refrigerants:
        filtered = filtered[filtered["Refrigerant"].isin(refrigerants)]
    return filtered


def render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Market Dashboard")
    st.markdown(
        "<p class='muted'>Explore how observed prices vary by brand, package type, refrigerant, power, and efficiency.</p>",
        unsafe_allow_html=True,
    )

    filtered = filtered_dashboard_data(df)
    if filtered.empty:
        st.warning("No offers match the current filters.")
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("Offers", f"{len(filtered):,}")
    metric_cols[1].metric("Manufacturers", filtered["Manufacturer display"].nunique())
    metric_cols[2].metric("Median price", format_eur(filtered["Price"].median()))
    metric_cols[3].metric("Median SCOP", format_number(filtered["SCOP"].median(), 2))
    metric_cols[4].metric("Median EUR/kW", format_eur(filtered["Price per kW"].median()))

    chart_cols = st.columns(2, gap="large")
    with chart_cols[0]:
        st.markdown("**Price vs. efficiency**")
        st.scatter_chart(
            filtered,
            x="SCOP",
            y="Price",
            color="Manufacturer display",
            size="Rated Power low T [kW]",
            height=360,
        )
    with chart_cols[1]:
        st.markdown("**Price vs. rated power**")
        st.scatter_chart(
            filtered,
            x="Rated Power low T [kW]",
            y="Price",
            color="Configuration",
            size="SCOP",
            height=360,
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
        st.markdown("**Package benchmarks**")
        st.dataframe(
            config_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "median_price": st.column_config.NumberColumn("Median price", format="EUR %.0f"),
                "median_scop": st.column_config.NumberColumn("Median SCOP", format="%.2f"),
            },
        )

    st.markdown("**Searchable matched offers**")
    query = st.text_input("Search manufacturer, model, title, or website", placeholder="e.g. Vaillant, R290, idealo")
    table = filtered.copy()
    if query:
        needle = query.lower()
        table = table[
            table[["Manufacturer display", "Model", "Titel", "Model/Type", "Website", "Refrigerant"]]
            .fillna("")
            .astype(str)
            .apply(lambda row: row.str.lower().str.contains(needle, regex=False).any(), axis=1)
        ]

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
    st.dataframe(
        table.sort_values("Price")[display_columns],
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
    st.subheader("Methodology")
    st.write(
        "The app uses the processed research dataset created from European online heat pump listings "
        "matched to technical product records. The offer checker benchmarks an entered price against "
        "observed prices for comparable products."
    )
    st.markdown(
        """
        **How the label is assigned**

        - `Good price`: at or below the 25th percentile of comparable observed offers
        - `Fair price`: between the 25th and 60th percentile
        - `Expensive`: between the 60th and 85th percentile
        - `Very expensive`: above the 85th percentile

        Comparable offers are selected by package type, refrigerant, manufacturer, rated power, and SCOP
        when enough observations exist. The app broadens the comparison group automatically when a very
        narrow group would be too small.
        """
    )
    st.info(
        "This is a research demo. It benchmarks product/package offer prices only. It should not be used as a final purchasing recommendation."
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

    tab_checker, tab_dashboard, tab_methodology = st.tabs(["Offer checker", "Dashboard", "Methodology"])
    with tab_checker:
        render_offer_checker(df)
    with tab_dashboard:
        render_dashboard(df)
    with tab_methodology:
        render_methodology(df)


if __name__ == "__main__":
    main()
