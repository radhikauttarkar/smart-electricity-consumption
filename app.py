
from io import BytesIO
from html import escape
import textwrap

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

st.set_page_config(
    page_title="Smart Energy",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = "Smart_Household_Electricity_Consumption_Dataset.csv"
TARGET = "Monthly_Electricity_Consumption_kWh"
CLS_FEATURES = [
    "Temperature_C", "Occupancy_Hours", "AC_Usage_Hours",
    "TV_Usage_Hours", "Water_Heater_Usage_Hours",
    "Computer_Usage_Hours",
]


def estimated_bill(consumption_kwh: float, tariff_per_kwh: float) -> float:
    return round(consumption_kwh, 2) * tariff_per_kwh


def render_result_card(
    label: str,
    value: str,
    unit: str,
    note: str = "",
) -> None:
    note_html = (
        f'<div class="result-card-note">{escape(note)}</div>' if note else ""
    )
    st.markdown(
        f'<div class="result-card">'
        f'<div class="result-card-label">{escape(label)}</div>'
        f'<div class="result-card-value">{escape(value)}</div>'
        f'<div class="result-card-unit">{escape(unit)}</div>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )


def create_energy_report_pdf(
    household: dict,
    prediction: float,
    category: int | None,
    tariff: float,
    appliance_results: pd.DataFrame,
    recommendations: list[str],
    model_metrics: dict,
    planned_usage: float,
    energy_savings: float,
    savings_percent: float,
    current_bill: float,
    planned_bill: float,
    bill_savings: float,
) -> bytes:
    report_buffer = BytesIO()

    with PdfPages(report_buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle(
            "Smart Energy Report",
            fontsize=18,
            fontweight="bold",
            color="#07583F",
            y=0.97,
        )
        fig.text(
            0.08, 0.935,
            "Your prediction and personalised energy-saving estimate",
            fontsize=10,
            color="#536B7D",
        )

        summary_ax = fig.add_axes([0.08, 0.69, 0.84, 0.20])
        summary_ax.axis("off")
        category_text = (
            "High" if category == 1 else
            "Low" if category == 0 else
            "Unavailable"
        )
        summary_rows = [
            ["Predicted monthly use", f"{prediction:,.2f} kWh"],
            ["Consumption category", category_text],
            ["Estimated bill before plan", f"Rs {current_bill:,.2f}/month"],
            ["Estimated bill after plan", f"Rs {planned_bill:,.2f}/month"],
            ["Bill formula", f"{prediction:,.2f} kWh x Rs {tariff:,.2f}/kWh"],
            ["Potential energy savings",
             f"{energy_savings:,.2f} kWh/month ({savings_percent:.1f}%)"],
            ["Potential bill savings", f"Rs {bill_savings:,.2f}/month"],
            ["Electricity tariff", f"Rs {tariff:,.2f} per kWh"],
            ["Estimated use after plan", f"{planned_usage:,.2f} kWh/month"],
        ]
        summary_table = summary_ax.table(
            cellText=summary_rows,
            colLabels=["Report metric", "Result"],
            cellLoc="left",
            colLoc="left",
            colWidths=[0.52, 0.48],
            bbox=[0, 0, 1, 1],
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(9)
        for (row, _), cell in summary_table.get_celld().items():
            cell.set_edgecolor("#DCE7E1")
            if row == 0:
                cell.set_facecolor("#E4F5EA")
                cell.set_text_props(weight="bold", color="#07583F")
            else:
                cell.set_facecolor("#FFFFFF" if row % 2 else "#F6FAF7")
                cell.set_text_props(color="#102F30")

        input_ax = fig.add_axes([0.08, 0.11, 0.84, 0.51])
        input_ax.axis("off")
        input_rows = [
            [str(name).replace("_", " "), str(value)]
            for name, value in household.items()
            if name != "Customer_ID"
        ]
        input_ax.set_title(
            "Your household details",
            loc="left",
            fontsize=11,
            color="#07583F",
            pad=8,
        )
        input_table = input_ax.table(
            cellText=input_rows,
            colLabels=["Input", "Value"],
            cellLoc="left",
            colLoc="left",
            colWidths=[0.58, 0.42],
            bbox=[0, 0, 1, 0.90],
        )
        input_table.auto_set_font_size(False)
        input_table.set_fontsize(8)
        for (row, _), cell in input_table.get_celld().items():
            cell.set_edgecolor("#E1EBE5")
            if row == 0:
                cell.set_facecolor("#E4F5EA")
                cell.set_text_props(weight="bold", color="#07583F")
            else:
                cell.set_facecolor("#FFFFFF" if row % 2 else "#F6FAF7")
                cell.set_text_props(color="#102F30")

        fig.text(
            0.08, 0.065,
            (
                f"Held-out MAE {model_metrics['MAE']:.1f} kWh | "
                f"RMSE {model_metrics['RMSE']:.1f} kWh | "
                f"R² {model_metrics['R²']:.3f}"
                    f" | Mean-baseline MAE {model_metrics['Baseline MAE']:.1f} | "
                    f"RMSE {model_metrics['Baseline RMSE']:.1f}"
                    + (
                    f" | High-class P/R/F1 "
                    f"{model_metrics['Precision']:.2f}/"
                    f"{model_metrics['Recall']:.2f}/"
                    f"{model_metrics['F1-score']:.2f}"
                    if "Precision" in model_metrics else ""
                )
            ),
            fontsize=7,
            color="#536B7D",
        )
        fig.text(
            0.08, 0.045,
            "Limits: no time-stamped monthly history is available; forecasts "
            "are seasonal scenarios, and savings use assumed appliance "
            "reductions.",
            fontsize=7,
            color="#536B7D",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle(
            "Your Before-and-After Plan",
            fontsize=17,
            fontweight="bold",
            color="#07583F",
            y=0.96,
        )
        usage_ax = fig.add_axes([0.12, 0.69, 0.76, 0.18])
        scenarios = ["Predicted\ncurrent", "After\noptimisation"]
        usage_bars = usage_ax.bar(
            scenarios,
            [prediction, planned_usage],
            color=["#72B99A", "#086548"],
            width=0.48,
        )
        usage_ax.set_title("Monthly electricity use", color="#07583F")
        usage_ax.set_ylabel("kWh/month")
        usage_ax.bar_label(usage_bars, fmt="%.1f", padding=3)
        usage_ax.grid(axis="y", alpha=0.2)
        usage_ax.spines[["top", "right"]].set_visible(False)

        savings_ax = fig.add_axes([0.18, 0.39, 0.72, 0.21])
        appliance_savings = appliance_results.sort_values(
            "Potential kWh difference", ascending=True
        )
        savings_bars = savings_ax.barh(
            appliance_savings["Appliance"],
            appliance_savings["Potential kWh difference"],
            color="#18815B",
        )
        savings_ax.set_title("Estimated monthly savings by appliance",
                             color="#07583F")
        savings_ax.set_xlabel("kWh saved/month")
        savings_ax.bar_label(savings_bars, fmt="%.1f", padding=3, fontsize=8)
        savings_ax.tick_params(axis="y", labelsize=8)
        savings_ax.grid(axis="x", alpha=0.2)
        savings_ax.spines[["top", "right"]].set_visible(False)

        recommendations_ax = fig.add_axes([0.10, 0.06, 0.82, 0.26])
        recommendations_ax.axis("off")
        recommendations_ax.text(
            0, 1.0, "Personalised appliance recommendations",
            fontsize=12, fontweight="bold", color="#07583F",
            va="top",
        )
        recommendation_text = "\n".join(
            f"- {textwrap.fill(item, width=92, subsequent_indent='  ')}"
            for item in recommendations
        )
        recommendations_ax.text(
            0, 0.86,
            recommendation_text,
            fontsize=8, color="#102F30", va="top", linespacing=1.25,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return report_buffer.getvalue()


def create_personalized_recommendations(
    household: dict,
    appliance_results: pd.DataFrame,
) -> list[str]:
    recommendations = []

    def quantity_label(value: float, singular: str) -> str:
        quantity = f"{value:g}"
        noun = singular if value == 1 else f"{singular}s"
        return f"{quantity} {noun}"

    for _, row in appliance_results.iterrows():
        appliance = row["Appliance"]
        saving = float(row["Potential kWh difference"])
        reduction = float(row["Suggested reduction (%)"])

        if appliance == "Air Conditioner":
            count = max(0.0, float(household.get("AC_Count", 0)))
            hours = max(0.0, float(household.get("AC_Usage_Hours", 0)))
            action = (
                f"You run {quantity_label(count, 'AC unit')} for about "
                f"{hours:g} hours/day. "
                "Set the thermostat to 24-26°C, use the sleep timer, and "
                "clean the filters monthly."
            )
        elif appliance == "Fans":
            count = max(0.0, float(household.get("Fan_Count", 0)))
            hours = max(0.0, float(household.get("Occupancy_Hours", 0)))
            action = (
                f"You use {quantity_label(count, 'fan')} for up to {hours:g} "
                "occupied hours/day. Switch them off in empty rooms and "
                "clean the blades to keep airflow effective."
            )
        elif appliance == "Lighting":
            count = max(0.0, float(household.get("Light_Count", 0)))
            hours = max(0.0, float(household.get("Occupancy_Hours", 0)))
            action = (
                f"You use {quantity_label(count, 'regular light')} during "
                f"{hours:g} occupied hours/day. Use daylight where possible "
                "and switch off lights when leaving a room."
            )
        elif appliance == "LED Lighting":
            count = max(0.0, float(household.get("LED_Bulbs_Count", 0)))
            hours = max(0.0, float(household.get("Occupancy_Hours", 0)))
            action = (
                f"You use {quantity_label(count, 'LED bulb')} during about "
                f"{hours:g} occupied hours/day. Keep only the lights needed "
                "for the task on, especially in shared spaces."
            )
        elif appliance == "Television":
            hours = max(0.0, float(household.get("TV_Usage_Hours", 0)))
            action = (
                f"Your TV is used about {hours:g} hours/day. Turn it fully "
                "off instead of leaving it on standby, and enable its "
                "auto-off timer."
            )
        elif appliance == "Water Heater":
            hours = max(
                0.0, float(household.get("Water_Heater_Usage_Hours", 0))
            )
            action = (
                f"Your water heater is used about {hours:g} hours/day. "
                "Use a timer and a moderate thermostat setting, and switch "
                "it off once hot water is ready."
            )
        elif appliance == "Computers":
            count = max(0.0, float(household.get("Computer_Count", 0)))
            hours = max(0.0, float(household.get("Computer_Usage_Hours", 0)))
            action = (
                f"You use {quantity_label(count, 'computer')} for about "
                f"{hours:g} "
                "hours/day. Enable power-saving sleep, and shut them down "
                "overnight or during longer breaks."
            )
        elif appliance == "Washing Machine":
            cycles = max(
                0.0,
                float(household.get("Washing_Machine_Usage_Per_Week", 0)),
            )
            action = (
                f"You run about {quantity_label(cycles, 'washing-machine cycle')} "
                "per "
                "week. Combine small loads into full loads and use the "
                "eco or cold-water cycle when suitable."
            )
        elif appliance == "Other Appliances":
            count = max(
                0.0, float(household.get("Other_Appliances_Count", 0))
            )
            hours = max(
                0.0,
                float(household.get("Other_Appliances_Usage_Hours", 0)),
            )
            action = (
                f"Your household uses "
                f"{quantity_label(count, 'other appliance')} for about "
                f"{hours:g} hours/day. Group their use when practical and "
                "switch off or unplug devices that are idle."
            )
        else:
            action = (
                "Keep the refrigerator door closed, check that its door "
                "seal is snug, and remove dust from the rear coils."
            )

        recommendations.append(
            f"{action} Following the suggested {reduction:.0f}% reduction "
            f"could save an estimated {saving:.1f} kWh/month."
        )

    return recommendations


def estimate_appliance_plan(
    household: dict,
    current_total: float,
) -> pd.DataFrame:
    appliance_specs = [
        ("Air Conditioner", "AC_Count", "AC_Usage_Hours", 1500, 0.20),
        ("Fans", "Fan_Count", "Occupancy_Hours", 75, 0.15),
        ("Lighting", "Light_Count", "Occupancy_Hours", 9, 0.10),
        ("LED Lighting", "LED_Bulbs_Count", "Occupancy_Hours", 9, 0.10),
        ("Television", None, "TV_Usage_Hours", 100, 0.20),
        ("Water Heater", "Water_Heater", "Water_Heater_Usage_Hours",
         2000, 0.15),
        ("Computers", "Computer_Count", "Computer_Usage_Hours", 100, 0.15),
        ("Washing Machine", None, "Washing_Machine_Usage_Per_Week",
         500, 0.15),
        ("Other Appliances", "Other_Appliances_Count",
         "Other_Appliances_Usage_Hours", 100, 0.10),
        ("Refrigerator", "Refrigerator", None, 150, 0.05),
    ]

    raw_rows = []
    for name, count_feature, hours_feature, watts, reduction in appliance_specs:
        count_value = household.get(count_feature, 1) if count_feature else 1
        if isinstance(count_value, str):
            count = (
                1 if count_value.strip().lower() in {"yes", "true", "1"}
                else 0
            )
        else:
            count = max(0.0, float(count_value))

        if name == "Washing Machine":
            hours_per_day = (
                max(0.0, float(household.get(hours_feature, 0.0)))
                * 0.5 / 7
            )
        elif name == "Refrigerator":
            hours_per_day = 8.0 if count else 0.0
        elif hours_feature:
            hours_per_day = max(
                0.0, float(household.get(hours_feature, 0.0))
            )
        else:
            hours_per_day = 0.0

        raw_rows.append({
            "Appliance": name,
            "Estimated current kWh/month": (
                count * watts / 1000 * hours_per_day * 30
            ),
            "Suggested reduction (%)": reduction * 100,
        })

    result = pd.DataFrame(raw_rows)
    modeled_total = float(result["Estimated current kWh/month"].sum())
    if modeled_total > 0:
        result["Current kWh/month"] = (
            result["Estimated current kWh/month"]
            * current_total / modeled_total
        )
    else:
        result["Current kWh/month"] = 0.0

    result["Planned kWh/month"] = (
        result["Current kWh/month"]
        * (1 - result["Suggested reduction (%)"] / 100)
    )
    result["Potential kWh difference"] = (
        result["Current kWh/month"] - result["Planned kWh/month"]
    )
    return result[result["Current kWh/month"] > 0].copy()


# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown("""
<style>
:root { color-scheme: light; }

html, body, .stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #FFFFFF !important;
    color: #102F30 !important;
}

[data-testid="stHeader"] {
    display: none !important;
}

body,
p,
label,
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
[data-testid="stExpander"] summary,
[data-testid="stAlert"] p,
[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    color: #102F30 !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #E7EEE9;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #38564D !important;
}

[data-testid="stCaptionContainer"] {
    color: #536B7D !important;
}

[data-baseweb="input"],
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #102F30 !important;
    border-color: #D7E3DC !important;
}

[data-testid="stSelectbox"] [role="combobox"] {
    background-color: #FFFFFF !important;
    color: #102F30 !important;
}

[data-testid="stNumberInput"] button,
[data-testid="stSelectbox"] button {
    color: #38564D !important;
    background-color: #FFFFFF !important;
}

input::placeholder,
textarea::placeholder {
    color: #647A70 !important;
    opacity: 1 !important;
}

[data-testid="stTabs"] [role="tablist"] {
    background: #FFFFFF !important;
    border-bottom: 1px solid #E1EBE5;
}

[data-testid="stTabs"] button {
    color: #536B7D !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #086548 !important;
    border-bottom-color: #086548 !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #FFFFFF !important;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border-color: #E1EBE5 !important;
}

div.stDownloadButton > button {
    background: #086548 !important;
    color: #FFFFFF !important;
    border: 1px solid #086548 !important;
    border-radius: 10px !important;
    min-height: 46px;
    font-weight: 650;
}

div.stDownloadButton > button * {
    color: #FFFFFF !important;
}

.block-container {
    max-width: 100% !important;
    padding: 3.5rem 5% 2rem !important;
}

.home-header {
    color: #07583F;
    font-size: 23px;
    font-weight: 750;
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 34px;
    height: 34px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #086548;
    color: #FFFFFF;
    font-size: 19px;
}

.home-tagline {
    color: #536779;
    text-align: right;
    font-size: 13px;
    padding-top: 9px;
}

.hero-title {
    color: #102F30;
    font-size: clamp(36px, 3.6vw, 55px);
    font-weight: 800;
    line-height: 1.17;
    letter-spacing: -1.5px;
}

.hero-description {
    color: #536B7D;
    font-size: 18px;
    line-height: 1.8;
    margin-top: 23px;
    margin-bottom: 25px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #086548 !important;
    color: #FFFFFF !important;
    border: 1px solid #086548 !important;
    border-radius: 10px !important;
    min-height: 50px;
    font-size: 16px;
    font-weight: 650;
}

div.stButton > button *,
div.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #064D38 !important;
    color: #FFFFFF !important;
}

div.stButton > button:hover *,
div.stFormSubmitButton > button:hover * {
    color: #FFFFFF !important;
}

[data-testid="stMetric"] {
    background: #F3FAF6;
    border: 1px solid #E1EFE7;
    padding: 16px;
    border-radius: 12px;
}

.result-card {
    min-height: 112px;
    padding: 14px 16px;
    background: #F3FAF6;
    border: 1px solid #E1EFE7;
    border-radius: 12px;
    overflow-wrap: anywhere;
}

.result-card-label {
    color: #38564D;
    font-size: 0.86rem;
    font-weight: 600;
    line-height: 1.3;
}

.result-card-value {
    color: #102F30;
    font-size: clamp(1.2rem, 2vw, 1.65rem);
    font-weight: 700;
    line-height: 1.25;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

.result-card-unit {
    color: #536B7D;
    font-size: 0.82rem;
    line-height: 1.3;
    margin-top: 2px;
}

.result-card-note {
    color: #536B7D;
    font-size: 0.73rem;
    line-height: 1.3;
    margin-top: 5px;
}

[data-testid="stImage"] img {
    max-width: 100%;
    height: auto;
}

[data-testid="stDownloadButton"] button {
    white-space: normal;
    overflow-wrap: anywhere;
}

.app-footer {
    margin-top: 3rem;
    padding: 1.25rem 0 1rem;
    border-top: 1px solid #E1EBE5;
    color: #536B7D;
    font-size: 0.88rem;
    line-height: 1.6;
}

[data-testid="stTabs"] button {
    color: #07583F !important;
}

.feature-section {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0;
    background: #F3FAF6;
    padding: 22px 4% 24px;
    margin: 0 -5.5% 0;
}

.feature-card {
    min-height: 145px;
    padding: 4px 20px;
    border-right: 1px solid #DFEBE4;
}

.feature-card:last-child {
    border-right: 0;
}

.feature-icon {
    width: 76px;
    height: 76px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #DFF2E7;
    color: #086548;
    font-size: 34px;
}

.feature-title {
    color: #07583F;
    font-size: 17px;
    font-weight: 700;
    text-align: center;
    margin-top: 14px;
}

.feature-description {
    color: #536B7D;
    font-size: 14px;
    line-height: 1.8;
    text-align: center;
    margin-top: 9px;
}

@media (max-width: 700px) {
    .hero-title { font-size: 36px; }
    .hero-description { font-size: 16px; }
    .feature-section { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .feature-card { padding: 12px; }
    .feature-card:nth-child(2) { border-right: 0; }
    .feature-card:nth-child(-n+2) { border-bottom: 1px solid #DFEBE4; }
    .result-card { min-height: 96px; padding: 12px; }
    .block-container { padding: 1.5rem 1rem 2.5rem !important; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LANDING PAGE ILLUSTRATION
# Rendered as HTML/SVG, NOT Markdown text.
# =========================================================
HOUSE_SVG = r"""
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 620 445" width="100%">
<defs>
 <linearGradient id="hill" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#F0F9F1"/>
  <stop offset="1" stop-color="#DDF2E4"/>
 </linearGradient>
 <linearGradient id="roof" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#147653"/>
  <stop offset="1" stop-color="#07563F"/>
 </linearGradient>
</defs>
<ellipse cx="320" cy="275" rx="278" ry="175" fill="url(#hill)"/>
<g fill="#DCEFE3">
 <ellipse cx="112" cy="194" rx="45" ry="22"/>
 <ellipse cx="145" cy="183" rx="29" ry="29"/>
 <ellipse cx="174" cy="198" rx="34" ry="18"/>
 <ellipse cx="470" cy="152" rx="37" ry="18"/>
 <ellipse cx="498" cy="142" rx="26" ry="25"/>
 <ellipse cx="525" cy="154" rx="32" ry="16"/>
</g>
<g stroke="#0B6A4D" stroke-width="6" stroke-linecap="round" fill="none">
 <path d="M310 42v-18 M260 67l-12-12 M360 67l12-12"/>
 <path d="M240 105h-18 M380 105h18"/>
</g>
<path d="M278 102a42 42 0 1 1 64 35c-7 7-9 13-10 20h-37
 c-1-8-4-14-10-21a41 41 0 0 1-7-34z"
 fill="#F7FFF9" stroke="#126A4E" stroke-width="5"/>
<path d="M294 127c9-24 27-27 38-29-2 22-13 39-32 42"
 fill="#18815B"/>
<path d="M296 141l17-27" stroke="#A5D7B8" stroke-width="3"/>
<path d="M295 159h31 M297 168h27 M301 177h19"
 stroke="#0D6A4D" stroke-width="7" stroke-linecap="round"/>
<path d="M310 192v8" stroke="#0D6A4D" stroke-width="5"/>
<!-- House -->
<path d="M145 285L248 174h218l99 111v148H145z" fill="#FFFDF8"/>
<path d="M132 285L248 164h219l108 121H132z" fill="url(#roof)"/>
<path d="M150 281L249 177" stroke="#0A503B" stroke-width="9"
 stroke-linecap="round"/>
<!-- Solar panels -->
<path d="M265 190l37 0 51 62h-38z
 M309 190h37l51 62h-37z
 M353 190h37l51 62h-37z"
 fill="#1D8061" stroke="#B7E2CC" stroke-width="3"/>
<rect x="455" y="146" width="32" height="43" rx="3" fill="#0B684B"/>
<rect x="274" y="330" width="45" height="103" rx="3" fill="#0B684B"/>
<circle cx="309" cy="381" r="3" fill="#B9E5CC"/>
<g fill="#76BCA2">
 <rect x="185" y="251" width="29" height="29" rx="2"/>
 <rect x="220" y="251" width="29" height="29" rx="2"/>
 <rect x="185" y="287" width="29" height="29" rx="2"/>
 <rect x="220" y="287" width="29" height="29" rx="2"/>
 <rect x="369" y="331" width="25" height="48" rx="2"/>
 <rect x="405" y="331" width="25" height="48" rx="2"/>
 <rect x="443" y="331" width="25" height="48" rx="2"/>
</g>
<!-- Tree and garden -->
<path d="M87 426V328" stroke="#0C7554" stroke-width="6"
 stroke-linecap="round"/>
<ellipse cx="87" cy="308" rx="25" ry="47" fill="#78B96A"/>
<path d="M87 328v54m0-19l-15-17m15 6l15-18"
 stroke="#0C7554" stroke-width="4" stroke-linecap="round" fill="none"/>
<g fill="#72B66C">
 <ellipse cx="57" cy="422" rx="33" ry="18"/>
 <ellipse cx="98" cy="424" rx="35" ry="18"/>
 <ellipse cx="480" cy="422" rx="35" ry="18"/>
 <ellipse cx="525" cy="422" rx="30" ry="17"/>
 <ellipse cx="355" cy="426" rx="30" ry="17"/>
 <ellipse cx="388" cy="424" rx="27" ry="15"/>
</g>
<path d="M50 438h525" stroke="#0C7554" stroke-width="5"
 stroke-linecap="round"/>
<!-- Energy monitor -->
<g transform="translate(500 267)">
 <rect width="76" height="118" rx="15" fill="#F9FFF9"
 stroke="#B8DDC4" stroke-width="7"/>
 <path d="M39 19l-14 28h13l-5 23 23-34H42l6-17z" fill="#0C7554"/>
 <rect x="19" y="83" width="8" height="13" rx="2" fill="#8CCBA4"/>
 <rect x="33" y="74" width="8" height="22" rx="2" fill="#65B88A"/>
 <rect x="47" y="64" width="8" height="32" rx="2" fill="#27875F"/>
</g>
</svg>
"""


# =========================================================
# HOME PAGE
# =========================================================
if "start_analysis" not in st.session_state:
    st.session_state.start_analysis = False

if not st.session_state.start_analysis:

    header_left, header_right = st.columns(2)

    with header_left:
        st.markdown(
            '<div class="home-header"><span class="brand-icon">ϟ</span>'
            'Smart Energy</div>',
            unsafe_allow_html=True,
        )

    with header_right:
        st.markdown(
            '<div class="home-tagline">'
            'Smarter Choices &nbsp; | &nbsp; Greener Tomorrow'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='border:0;border-top:1px solid #DCE7E1;"
        "margin-top:16px;margin-bottom:20px;'>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("""
        <div style="padding-top:0">
          <div class="hero-title" style="margin-top:0">
            Smart Electricity<br>
            Consumption &amp;<br>
            Optimisation
          </div>
          <div class="hero-description">
            Predict your household electricity usage, identify high
            consumption patterns and get personalised suggestions
            to reduce usage and save on your electricity bill.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "⚡  Start Analysis   →", key="start_home", width="content"
        ):
            st.session_state.start_analysis = True
            st.session_state["dashboard_tab"] = "🔮 Prediction"
            st.rerun()

    with right:
        components.html(
            HOUSE_SVG,
            height=420,
            scrolling=False,
        )

    features = [
        ("⌂", "Predict Consumption",
         "Estimate your household's monthly electricity use."),
        ("▥", "Find Usage Category",
         "See whether your estimated electricity use is low or high."),
        ("♧", "Optimise Your Energy",
         "Get practical tips and estimate potential energy and bill savings."),
        ("▤", "View Final Report",
         "Compare your current use with a savings plan and download a report."),
    ]

    feature_cards = "".join(
        f'<div class="feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-description">{description}</div>'
        f'</div>'
        for icon, title, description in features
    )
    st.markdown(
        f'<div class="feature-section">{feature_cards}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Smart Energy | Use electricity wisely.")
    st.stop()


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip()
    return data


try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.info(
        f"Place {DATA_FILE} in the same folder as app.py."
    )
    st.stop()

if TARGET not in df.columns:
    st.error(f"Required target column '{TARGET}' was not found.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df = df[np.isfinite(df[TARGET])].copy()

if len(df) < 10:
    st.error("At least 10 valid dataset records are required.")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown(
    '<div style="font-size:1.75rem;font-weight:700;color:#07583F;'
    'margin:0 0 1rem 0;">📊 Dashboard</div>',
    unsafe_allow_html=True,
)

tariff = st.sidebar.number_input(
    "Electricity tariff (₹ per kWh)",
    min_value=0.0,
    value=8.0,
    step=0.5,
)

st.sidebar.caption(
    "Bill estimates use a flat rate. Actual bills can include "
    "fixed charges, slabs and taxes."
)

if st.sidebar.button("← Back to Home"):
    st.session_state.start_analysis = False
    st.rerun()


# =========================================================
# REGRESSION MODEL
# =========================================================
regression_features = [
    c for c in df.columns if c not in [TARGET, "Customer_ID"]
]

if not regression_features:
    st.error("No usable prediction features found.")
    st.stop()

X_reg = df[regression_features].copy()
X_reg = X_reg.replace([np.inf, -np.inf], np.nan)
y_reg = df[TARGET].copy()

regression_features = [
    c for c in regression_features if X_reg[c].notna().any()
]
X_reg = X_reg[regression_features]

numeric_features = X_reg.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c for c in regression_features if c not in numeric_features
]

transformers = []

if numeric_features:
    transformers.append((
        "numeric",
        Pipeline([("imputer", SimpleImputer(strategy="median"))]),
        numeric_features,
    ))

if categorical_features:
    transformers.append((
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ))

preprocessor = ColumnTransformer(transformers=transformers)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
regression_evaluation_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression()),
])
regression_evaluation_model.fit(X_reg_train, y_reg_train)
regression_test_predictions = regression_evaluation_model.predict(X_reg_test)
regression_metrics = {
    "MAE": float(mean_absolute_error(y_reg_test, regression_test_predictions)),
    "RMSE": float(np.sqrt(mean_squared_error(
        y_reg_test, regression_test_predictions
    ))),
    "R²": float(r2_score(y_reg_test, regression_test_predictions)),
}
regression_baseline = DummyRegressor(strategy="mean")
regression_baseline.fit(X_reg_train, y_reg_train)
regression_baseline_predictions = regression_baseline.predict(X_reg_test)
regression_baseline_metrics = {
    "MAE": float(mean_absolute_error(
        y_reg_test, regression_baseline_predictions
    )),
    "RMSE": float(np.sqrt(mean_squared_error(
        y_reg_test, regression_baseline_predictions
    ))),
    "R²": float(r2_score(y_reg_test, regression_baseline_predictions)),
}

regression_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression()),
])
regression_model.fit(X_reg, y_reg)


# =========================================================
# CLASSIFICATION MODEL
# =========================================================
missing_cls = [c for c in CLS_FEATURES if c not in df.columns]
classifier_available = not missing_cls

if classifier_available:
    X_cls = df[CLS_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    valid = X_cls.notna().any(axis=1)
    X_cls = X_cls.loc[valid]
    y_cls_raw = df.loc[valid, TARGET]

    if len(X_cls) < 10:
        classifier_available = False

if classifier_available:
    X_cls_train, X_cls_test, y_cls_train_raw, y_cls_test_raw = (
        train_test_split(
            X_cls, y_cls_raw, test_size=0.2, random_state=42
        )
    )
    classification_threshold = float(y_cls_train_raw.median())
    y_cls_train = (y_cls_train_raw >= classification_threshold).astype(int)
    y_cls_test = (y_cls_test_raw >= classification_threshold).astype(int)
    y_cls = (y_cls_raw >= y_cls_raw.median()).astype(int)

    if y_cls_train.nunique() < 2 or y_cls.nunique() < 2:
        classifier_available = False

if classifier_available:
    classifier_evaluation_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    classifier_evaluation_model.fit(X_cls_train, y_cls_train)
    classification_test_predictions = classifier_evaluation_model.predict(
        X_cls_test
    )
    classification_metrics = {
        "Precision": float(precision_score(
            y_cls_test, classification_test_predictions,
            pos_label=1, zero_division=0,
        )),
        "Recall": float(recall_score(
            y_cls_test, classification_test_predictions,
            pos_label=1, zero_division=0,
        )),
        "F1-score": float(f1_score(
            y_cls_test, classification_test_predictions,
            pos_label=1, zero_division=0,
        )),
        "Confusion matrix": confusion_matrix(
            y_cls_test, classification_test_predictions, labels=[0, 1]
        ),
    }
    classifier_baseline = DummyClassifier(strategy="prior")
    classifier_baseline.fit(X_cls_train, y_cls_train)
    baseline_class_predictions = classifier_baseline.predict(X_cls_test)
    classification_baseline_metrics = {
        "Precision": float(precision_score(
            y_cls_test, baseline_class_predictions,
            pos_label=1, zero_division=0,
        )),
        "Recall": float(recall_score(
            y_cls_test, baseline_class_predictions,
            pos_label=1, zero_division=0,
        )),
        "F1-score": float(f1_score(
            y_cls_test, baseline_class_predictions,
            pos_label=1, zero_division=0,
        )),
    }

    cls_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    cls_model.fit(X_cls, y_cls)


# =========================================================
# DASHBOARD TABS
# =========================================================
def open_optimization_tab():
    st.session_state["dashboard_tab"] = "⚡ Optimisation"


if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = []

(
    predict_tab,
    optimization_tab,
    forecast_tab,
    goals_tab,
    model_tab,
    scenarios_tab,
) = (
    st.tabs([
        "🔮 Prediction",
        "⚡ Optimisation",
        "📈 Forecast",
        "🎯 Goals & What-if",
        "🧪 Model",
        "📂 Scenarios",
    ], key="dashboard_tab", on_change="rerun")
)

# =========================================================
# PREDICTION
# =========================================================
with predict_tab:
    st.subheader("Household Input & Consumption Prediction")
    st.write(
        "Enter household details once to estimate monthly electricity use "
        "and see whether the predicted consumption is Low or High."
    )

    with st.form("prediction_form"):
        input_values = {}
        input_cols = st.columns(2)

        for i, col in enumerate(regression_features):
            with input_cols[i % 2]:
                if col in numeric_features:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    default = vals.median()
                    if pd.isna(default):
                        default = 0.0
                    input_values[col] = st.number_input(
                        col, value=float(default), key=f"pred_{col}"
                    )
                else:
                    choices = sorted(
                        df[col].dropna().astype(str).unique().tolist()
                    )
                    if choices:
                        input_values[col] = st.selectbox(
                            col, choices, key=f"pred_{col}"
                        )
                    else:
                        input_values[col] = st.text_input(
                            col, key=f"pred_{col}"
                        )

        submitted = st.form_submit_button(
            "Predict Consumption", type="primary"
        )

    if submitted:
        input_df = pd.DataFrame(
            [input_values], columns=regression_features
        )

        for col in numeric_features:
            input_df[col] = pd.to_numeric(
                input_df[col], errors="coerce"
            )

        prediction = max(
            0.0, float(regression_model.predict(input_df)[0])
        )
        st.session_state.last_prediction = prediction
        st.session_state.last_input_values = input_values.copy()
        st.session_state.last_category = None
        if classifier_available:
            classification_input = pd.DataFrame(
                [{
                    col: pd.to_numeric(
                        pd.Series([input_values[col]]), errors="coerce"
                    ).iloc[0]
                    for col in CLS_FEATURES
                }],
                columns=CLS_FEATURES,
            )
            st.session_state.last_category = int(
                cls_model.predict(classification_input)[0]
            )

    if "last_prediction" in st.session_state:
        prediction = st.session_state.last_prediction
        p1, p2 = st.columns(2)
        with p1:
            render_result_card(
                "Monthly use", f"{prediction:,.2f}", "kWh / month"
            )
        with p2:
            render_result_card(
                "Estimated bill",
                f"₹{estimated_bill(prediction, tariff):,.2f}",
                "per month",
                f"{prediction:,.2f} kWh × ₹{tariff:,.2f}/kWh; "
                "excludes fixed charges and slabs",
            )

        category = st.session_state.last_category
        if category is not None:
            if category == 1:
                st.warning("Predicted Consumption Category: HIGH")
            else:
                st.success("Predicted Consumption Category: LOW")
        else:
            st.info("Low/High classification is not available for these details.")
        st.markdown("### Ready to reduce your electricity use?")
        st.write(
            "Continue to optimisation to compare your current appliance "
            "usage with a lower-use plan and estimate potential savings."
        )
        st.button(
            "⚡ Continue to Energy Optimisation",
            key="continue_to_optimization",
            type="primary",
            on_click=open_optimization_tab,
        )


# =========================================================
# ENERGY OPTIMISATION
# =========================================================
with optimization_tab:
    st.subheader("⚡ Smart Energy Optimisation")
    if "last_prediction" not in st.session_state:
        st.info(
            "Enter your household details in the Prediction tab and submit "
            "them first. The optimisation report will reuse those same inputs."
        )
        if st.button("Go to Prediction", key="optimization_to_prediction"):
            st.session_state["dashboard_tab"] = "🔮 Prediction"
            st.rerun()
    else:
        st.write(
            "This plan uses the household details you already entered for "
            "your prediction. No second appliance form is needed."
        )
        household = st.session_state.last_input_values
        current_total = float(st.session_state.last_prediction)
        result_df = estimate_appliance_plan(household, current_total)

        if result_df.empty:
            st.warning(
                "No appliance usage was available from the prediction inputs "
                "to estimate an optimisation plan."
            )
        else:
            recommendations = create_personalized_recommendations(
                household, result_df
            )
            planned_total = float(result_df["Planned kWh/month"].sum())
            difference = current_total - planned_total
            current_bill = estimated_bill(current_total, tariff)
            planned_bill = estimated_bill(planned_total, tariff)
            bill_difference = current_bill - planned_bill
            pct = difference / current_total * 100 if current_total else 0.0

            st.caption(
                "The model's predicted monthly consumption is the current "
                "baseline. Appliance shares are estimated from the usage and "
                "counts you entered; suggested reductions are planning "
                "assumptions, not guaranteed savings."
            )
            st.subheader("Your Optimisation Report")
            summary_cards = [
                ("Before plan", f"{current_total:,.2f}", "kWh / month", ""),
                ("After plan", f"{planned_total:,.2f}", "kWh / month", ""),
                (
                    "Energy saved", f"{difference:,.2f}", "kWh / month",
                    f"{pct:.1f}% reduction",
                ),
                (
                    "Bill before", f"₹{current_bill:,.2f}", "per month",
                    f"at ₹{tariff:,.2f}/kWh",
                ),
                (
                    "Bill after", f"₹{planned_bill:,.2f}", "per month",
                    f"at ₹{tariff:,.2f}/kWh",
                ),
                (
                    "Bill saved", f"₹{bill_difference:,.2f}", "per month",
                    f"at ₹{tariff:,.2f}/kWh",
                ),
            ]
            for start in range(0, len(summary_cards), 2):
                card_columns = st.columns(2)
                for column, card in zip(
                    card_columns, summary_cards[start:start + 2]
                ):
                    with column:
                        render_result_card(*card)

            st.success(
                f"Estimated bill formula: monthly kWh × tariff "
                f"(₹{tariff:,.2f}/kWh). Both before and after bills use this "
                f"same tariff and the displayed kWh rounded to 2 decimals."
            )

            st.subheader("Appliance-wise Estimated Comparison")
            st.dataframe(
                result_df[[
                    "Appliance", "Suggested reduction (%)",
                    "Current kWh/month", "Planned kWh/month",
                    "Potential kWh difference",
                ]].round(2),
                width="stretch",
                hide_index=True,
            )

            st.subheader("Monthly Usage and Bill Comparison")
            usage_column, bill_column = st.columns(2)
            scenarios = ["Predicted\ncurrent", "After\noptimisation"]
            usage_fig, usage_ax = plt.subplots(figsize=(5.2, 3.2))
            usage_bars = usage_ax.bar(
                scenarios, [current_total, planned_total],
                color=["#72B99A", "#086548"],
                width=0.58,
            )
            usage_ax.set_title("Monthly usage", fontsize=10)
            usage_ax.set_ylabel("kWh/month", fontsize=8)
            usage_ax.tick_params(axis="both", labelsize=8)
            usage_ax.bar_label(
                usage_bars, fmt="%.1f", padding=2, fontsize=8
            )
            usage_ax.grid(axis="y", alpha=0.2)
            usage_ax.spines[["top", "right"]].set_visible(False)
            usage_fig.tight_layout()
            with usage_column:
                st.pyplot(usage_fig, width="stretch")
            plt.close(usage_fig)

            bill_fig, bill_ax = plt.subplots(figsize=(5.2, 3.2))
            bill_bars = bill_ax.bar(
                scenarios, [current_bill, planned_bill],
                color=["#72B99A", "#086548"],
                width=0.58,
            )
            bill_ax.set_title("Estimated monthly bill", fontsize=10)
            bill_ax.set_ylabel("₹/month", fontsize=8)
            bill_ax.tick_params(axis="both", labelsize=8)
            bill_ax.bar_label(bill_bars, fmt="%.0f", padding=2, fontsize=8)
            bill_ax.grid(axis="y", alpha=0.2)
            bill_ax.spines[["top", "right"]].set_visible(False)
            bill_fig.tight_layout()
            with bill_column:
                st.pyplot(bill_fig, width="stretch")
            plt.close(bill_fig)

            st.subheader("Estimated Savings by Appliance")
            savings_chart, savings_ax = plt.subplots(figsize=(6.2, 3.0))
            savings_data = result_df.sort_values(
                "Potential kWh difference", ascending=True
            )
            savings_bars = savings_ax.barh(
                savings_data["Appliance"],
                savings_data["Potential kWh difference"],
                color="#18815B",
            )
            savings_ax.set_xlabel("kWh saved per month", fontsize=8)
            savings_ax.tick_params(axis="both", labelsize=8)
            savings_ax.bar_label(
                savings_bars, fmt="%.1f", padding=2, fontsize=8
            )
            savings_ax.grid(axis="x", alpha=0.2)
            savings_ax.spines[["top", "right"]].set_visible(False)
            savings_chart.tight_layout()
            st.pyplot(savings_chart, width="content")
            plt.close(savings_chart)

            with st.expander("Personalised appliance recommendations"):
                for recommendation in recommendations:
                    st.write(f"- {recommendation}")

            scenario_name = st.text_input(
                "Name this analysis",
                value=f"Scenario {len(st.session_state.saved_scenarios) + 1}",
                key="scenario_name",
            )
            if st.button("Save analysis for comparison"):
                st.session_state.saved_scenarios.append({
                    "Name": scenario_name.strip() or (
                        f"Scenario {len(st.session_state.saved_scenarios) + 1}"
                    ),
                    "Saved at": pd.Timestamp.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Baseline usage (kWh/month)": current_total,
                    "Scenario usage (kWh/month)": planned_total,
                    "Usage change (kWh/month)": difference,
                    "Baseline bill": current_bill,
                    "Scenario bill": planned_bill,
                    "Bill change": bill_difference,
                    "Household": household.copy(),
                })
                st.success("Analysis saved in this session.")

            csv_rows = [
                {"Section": "Prediction", "Item": "Monthly use (kWh)",
                 "Value": current_total},
                {"Section": "Optimisation", "Item": "Planned use (kWh)",
                 "Value": planned_total},
                {"Section": "Optimisation", "Item": "Energy savings (kWh)",
                 "Value": difference},
                {"Section": "Optimisation", "Item": "Current bill",
                 "Value": current_bill},
                {"Section": "Optimisation", "Item": "Planned bill",
                 "Value": planned_bill},
                {"Section": "Optimisation", "Item": "Bill savings",
                 "Value": bill_difference},
                {"Section": "Model limitations", "Item": "Assumptions",
                 "Value": (
                     "Dataset has no time-stamped monthly history; forecasts "
                     "are seasonal scenarios. Appliance savings use assumed "
                     "reductions and are estimates."
                 )},
            ]
            csv_rows.extend(
                {"Section": "Household input",
                 "Item": str(name).replace("_", " "), "Value": value}
                for name, value in household.items()
            )
            csv_rows.extend(
                {"Section": "Appliance estimate",
                 "Item": row["Appliance"],
                 "Value": (
                     f"Current {row['Current kWh/month']:.2f} kWh/month; "
                     f"planned {row['Planned kWh/month']:.2f}; "
                     f"saving {row['Potential kWh difference']:.2f}"
                 )}
                for _, row in result_df.iterrows()
            )
            csv_rows.extend(
                {"Section": "Recommendation",
                 "Item": f"Recommendation {index}",
                 "Value": recommendation}
                for index, recommendation in enumerate(recommendations, 1)
            )
            csv_rows.extend(
                {"Section": "Held-out model performance",
                 "Item": metric, "Value": value}
                for metric, value in regression_metrics.items()
            )
            csv_rows.extend(
                {"Section": "Held-out mean baseline",
                 "Item": metric, "Value": value}
                for metric, value in regression_baseline_metrics.items()
            )
            if classifier_available:
                csv_rows.extend(
                    {"Section": "Held-out classification performance",
                     "Item": metric, "Value": value}
                    for metric, value in classification_metrics.items()
                    if metric != "Confusion matrix"
                )
            csv_report = pd.DataFrame(csv_rows).to_csv(index=False).encode(
                "utf-8"
            )
            st.download_button(
                "Download analysis CSV",
                data=csv_report,
                file_name="smart_energy_analysis.csv",
                mime="text/csv",
            )

            report_metrics = regression_metrics.copy()
            report_metrics.update({
                f"Baseline {metric}": value
                for metric, value in regression_baseline_metrics.items()
            })
            if classifier_available:
                report_metrics.update({
                    metric: value
                    for metric, value in classification_metrics.items()
                    if metric != "Confusion matrix"
                })
            pdf_report = create_energy_report_pdf(
                household=household,
                prediction=current_total,
                category=st.session_state.last_category,
                tariff=tariff,
                appliance_results=result_df,
                recommendations=recommendations,
                model_metrics=report_metrics,
                planned_usage=planned_total,
                energy_savings=difference,
                savings_percent=pct,
                current_bill=current_bill,
                planned_bill=planned_bill,
                bill_savings=bill_difference,
            )
            st.download_button(
                "Download Prediction & Optimisation PDF Report",
                data=pdf_report,
                file_name="smart_energy_prediction_optimisation_report.pdf",
                mime="application/pdf",
            )


with forecast_tab:
    st.subheader("Monthly consumption forecast")
    st.write(
        "Explore how the same household profile may vary by season over "
        "the coming months."
    )
    if "last_prediction" not in st.session_state:
        st.info("Run a prediction first to create a household-specific forecast.")
    else:
        household = st.session_state.last_input_values
        horizon = st.slider(
            "Forecast horizon (months)", min_value=3, max_value=12,
            value=6, key="forecast_horizon",
        )
        season_values = (
            sorted(df["Season"].dropna().astype(str).unique())
            if "Season" in df.columns else []
        )
        month_seasons = {
            1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer",
            5: "Summer", 6: "Summer", 7: "Monsoon", 8: "Monsoon",
            9: "Monsoon", 10: "Monsoon", 11: "Winter", 12: "Winter",
        }
        start_month = (
            pd.Timestamp.today().to_period("M").to_timestamp()
            + pd.offsets.MonthBegin(1)
        )
        forecast_dates = pd.date_range(
            start=start_month, periods=horizon, freq="MS"
        )
        forecast_rows = []
        for forecast_date in forecast_dates:
            profile = household.copy()
            requested_season = month_seasons[forecast_date.month]
            matched_season = next(
                (
                    value for value in season_values
                    if value.casefold() == requested_season.casefold()
                ),
                None,
            )
            if matched_season and "Season" in profile:
                profile["Season"] = matched_season
            forecast_input = pd.DataFrame(
                [profile], columns=regression_features
            )
            for feature in numeric_features:
                forecast_input[feature] = pd.to_numeric(
                    forecast_input[feature], errors="coerce"
                )
            usage = max(
                0.0, float(regression_model.predict(forecast_input)[0])
            )
            forecast_rows.append({
                "Month": forecast_date.strftime("%b %Y"),
                "Season": matched_season or "Profile season",
                "Forecast (kWh)": usage,
                "Lower estimate (kWh)": max(
                    0.0, usage - regression_metrics["RMSE"]
                ),
                "Upper estimate (kWh)": usage + regression_metrics["RMSE"],
            })

        forecast_df = pd.DataFrame(forecast_rows)
        forecast_chart, forecast_ax = plt.subplots(figsize=(8, 3.5))
        x_values = np.arange(len(forecast_df))
        forecast_ax.plot(
            x_values, forecast_df["Forecast (kWh)"],
            marker="o", color="#086548", label="Seasonal estimate",
        )
        forecast_ax.fill_between(
            x_values,
            forecast_df["Lower estimate (kWh)"],
            forecast_df["Upper estimate (kWh)"],
            color="#72B99A", alpha=0.24,
            label="± held-out RMSE (uncertainty range)",
        )
        forecast_ax.set_xticks(x_values)
        forecast_ax.set_xticklabels(forecast_df["Month"], rotation=30)
        forecast_ax.set_ylabel("kWh/month")
        forecast_ax.grid(axis="y", alpha=0.2)
        forecast_ax.spines[["top", "right"]].set_visible(False)
        forecast_ax.legend(frameon=False)
        forecast_chart.tight_layout()
        st.pyplot(forecast_chart, width="stretch")
        plt.close(forecast_chart)
        st.dataframe(
            forecast_df.round(1), hide_index=True, width="stretch"
        )
        st.caption(
            "The dataset contains household records and seasonal labels, but "
            "no dated monthly history. This is a seasonal scenario forecast "
            "for your entered profile, not a time-series trend or guarantee. "
            "The shaded range uses held-out test RMSE as a rough uncertainty "
            "indicator, not a statistical prediction interval."
        )


with goals_tab:
    st.subheader("Monthly savings goal")
    if "last_prediction" not in st.session_state:
        st.info("Run a prediction first to set a household-specific goal.")
    else:
        current_total = float(st.session_state.last_prediction)
        household = st.session_state.last_input_values
        goal_plan = estimate_appliance_plan(household, current_total)
        plan_usage = float(goal_plan["Planned kWh/month"].sum())
        goal_choice = st.radio(
            "Choose a monthly target",
            ["Reduce usage by a percentage", "Stay below a bill amount"],
            horizontal=True,
            key="goal_type",
        )
        if goal_choice == "Reduce usage by a percentage":
            target_percent = st.slider(
                "Consumption reduction target (%)",
                min_value=1, max_value=50, value=10, key="goal_percent",
            )
            target_usage = current_total * (1 - target_percent / 100)
            goal_description = (
                f"Use no more than {target_usage:,.1f} kWh/month "
                f"({target_percent}% below your prediction)."
            )
        else:
            target_bill = st.number_input(
                "Maximum monthly bill (₹)",
                min_value=0.0,
                value=max(0.0, current_total * tariff * 0.9),
                step=100.0,
                key="goal_bill",
            )
            target_usage = (
                target_bill / tariff if tariff > 0 else current_total
            )
            goal_description = (
                f"Keep the estimated monthly bill at or below "
                f"₹{target_bill:,.0f}."
            )

        required_savings = max(0.0, current_total - target_usage)
        planned_savings = max(0.0, current_total - plan_usage)
        goal_progress = (
            1.0 if required_savings == 0
            else min(1.0, planned_savings / required_savings)
        )
        st.write(goal_description)
        st.progress(goal_progress)
        if plan_usage <= target_usage:
            st.success("Your current optimisation plan meets this goal.")
        else:
            st.info(
                f"The current plan reaches {goal_progress:.0%} of your goal. "
                f"Estimated usage is {plan_usage:,.1f} kWh/month; "
                f"target is {target_usage:,.1f} kWh/month."
            )
        st.caption(
            "Progress compares the estimated appliance plan with your target. "
            "Savings rely on the optimisation assumptions and are not guaranteed."
        )

    st.divider()
    st.subheader("What-if simulator")
    if "last_prediction" not in st.session_state:
        st.info("Run a prediction to simulate changes to your household habits.")
    else:
        household = st.session_state.last_input_values
        what_if = household.copy()
        slider_columns = st.columns(2)
        adjustable_hours = [
            ("AC_Usage_Hours", "AC use (hours/day)"),
            ("Occupancy_Hours", "Fan and lighting active time (hours/day)"),
            ("Other_Appliances_Usage_Hours",
             "Other appliances use (hours/day)"),
        ]
        for index, (feature, label) in enumerate(adjustable_hours):
            if feature in what_if:
                default_value = float(np.clip(
                    float(what_if[feature]), 0.0, 24.0
                ))
                with slider_columns[index % 2]:
                    what_if[feature] = st.slider(
                        label, 0.0, 24.0, default_value, 0.5,
                        key=f"what_if_{feature}",
                    )

        adjustable_counts = [
            ("Light_Count", "Regular light count"),
            ("LED_Bulbs_Count", "LED bulb count"),
            ("Other_Appliances_Count", "Other appliance count"),
        ]
        for index, (feature, label) in enumerate(adjustable_counts):
            if feature in what_if:
                default_value = int(np.clip(
                    round(float(what_if[feature])), 0, 50
                ))
                with slider_columns[index % 2]:
                    what_if[feature] = st.slider(
                        label, 0, 50, default_value, 1,
                        key=f"what_if_{feature}",
                    )

        what_if_input = pd.DataFrame(
            [what_if], columns=regression_features
        )
        for feature in numeric_features:
            what_if_input[feature] = pd.to_numeric(
                what_if_input[feature], errors="coerce"
            )
        what_if_usage = max(
            0.0, float(regression_model.predict(what_if_input)[0])
        )
        current_usage = float(st.session_state.last_prediction)
        before_bill = estimated_bill(current_usage, tariff)
        after_bill = estimated_bill(what_if_usage, tariff)
        before_metric, after_metric = st.columns(2)
        with before_metric:
            render_result_card(
                "Current use", f"{current_usage:,.2f}", "kWh / month"
            )
        with after_metric:
            render_result_card(
                "What-if use",
                f"{what_if_usage:,.2f}",
                "kWh / month",
                f"Change: {what_if_usage - current_usage:+,.1f} kWh",
            )
        bill_before_metric, bill_after_metric = st.columns(2)
        with bill_before_metric:
            render_result_card(
                "Current bill", f"₹{before_bill:,.2f}", "per month",
                f"at ₹{tariff:,.2f}/kWh",
            )
        with bill_after_metric:
            render_result_card(
                "What-if bill", f"₹{after_bill:,.2f}", "per month",
                f"Change: ₹{after_bill - before_bill:+,.2f} · "
                f"at ₹{tariff:,.2f}/kWh",
            )
        st.caption(
            f"The simulator re-runs the model with the changed inputs. Bill = "
            f"estimated kWh × Dashboard tariff (₹{tariff:,.2f}/kWh); actual "
            "tariffs and fixed charges may differ."
        )
        what_if_name = st.text_input(
            "Name this what-if scenario",
            value=f"What-if {len(st.session_state.saved_scenarios) + 1}",
            key="what_if_scenario_name",
        )
        if st.button("Save what-if scenario"):
            st.session_state.saved_scenarios.append({
                "Name": what_if_name.strip() or (
                    f"What-if {len(st.session_state.saved_scenarios) + 1}"
                ),
                "Saved at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Baseline usage (kWh/month)": current_usage,
                "Scenario usage (kWh/month)": what_if_usage,
                "Usage change (kWh/month)": current_usage - what_if_usage,
                "Baseline bill": before_bill,
                "Scenario bill": after_bill,
                "Bill change": before_bill - after_bill,
                "Household": what_if.copy(),
            })
            st.success("What-if scenario saved in this session.")


with model_tab:
    st.subheader("Model performance")
    st.write(
        "Metrics below use a held-out 20% test split (80% train / 20% test). "
        "The regression model is compared with a mean-prediction baseline; "
        "classification is compared with a majority-class baseline. "
        "Prediction models are then refit on all available records."
    )
    st.markdown("#### Monthly consumption regression")
    regression_comparison = pd.DataFrame(
        {
            "Linear regression": regression_metrics,
            "Mean baseline": regression_baseline_metrics,
        }
    )
    regression_comparison.loc["MAE improvement vs baseline"] = (
        1 - regression_metrics["MAE"] / regression_baseline_metrics["MAE"]
        if regression_baseline_metrics["MAE"] else 0.0
    )
    regression_comparison.loc["RMSE improvement vs baseline"] = (
        1 - regression_metrics["RMSE"] / regression_baseline_metrics["RMSE"]
        if regression_baseline_metrics["RMSE"] else 0.0
    )
    st.dataframe(
        regression_comparison.style.format({
            "Linear regression": "{:.3f}",
            "Mean baseline": "{:.3f}",
        }).format({
            "Linear regression": "{:.1%}",
            "Mean baseline": "{:.1%}",
        }, subset=pd.IndexSlice[
            ["MAE improvement vs baseline", "RMSE improvement vs baseline"],
            :
        ]),
        width="stretch",
    )
    if classifier_available:
        st.markdown("#### Low / high consumption classification")
        st.caption(
            "The positive class is High consumption, defined using the "
            "training split's median monthly usage."
        )
        classification_comparison = pd.DataFrame({
            "Logistic regression": {
                name: classification_metrics[name]
                for name in ("Precision", "Recall", "F1-score")
            },
            "Majority baseline": classification_baseline_metrics,
        })
        st.dataframe(
            classification_comparison.style.format("{:.3f}"),
            width="stretch",
        )
        st.caption("Confusion matrix: rows are actual; columns are predicted.")
        st.dataframe(
            pd.DataFrame(
                classification_metrics["Confusion matrix"],
                index=["Actual Low", "Actual High"],
                columns=["Predicted Low", "Predicted High"],
            ),
            width="stretch",
        )
    else:
        st.info(
            "Classification metrics are unavailable because valid "
            "two-class test data could not be built."
        )
    st.caption(
        "Scores describe this dataset and split only. They are not a "
        "guarantee of accuracy for every household."
    )


with scenarios_tab:
    st.subheader("Saved analyses")
    st.write("Saved scenarios remain available for this browser session only.")
    if not st.session_state.saved_scenarios:
        st.info(
            "Save an analysis from Optimisation to compare household "
            "scenarios here."
        )
    else:
        scenario_df = pd.DataFrame(st.session_state.saved_scenarios)
        display_columns = [
            "Name", "Saved at", "Baseline usage (kWh/month)",
            "Scenario usage (kWh/month)", "Usage change (kWh/month)",
            "Baseline bill", "Scenario bill", "Bill change",
        ]
        st.dataframe(
            scenario_df[display_columns].round(2),
            hide_index=True,
            width="stretch",
        )
        comparison_chart, comparison_ax = plt.subplots(
            figsize=(8, 3.5)
        )
        positions = np.arange(len(scenario_df))
        width = 0.36
        comparison_ax.bar(
            positions - width / 2,
            scenario_df["Baseline usage (kWh/month)"],
            width,
            label="Baseline",
            color="#72B99A",
        )
        comparison_ax.bar(
            positions + width / 2,
            scenario_df["Scenario usage (kWh/month)"],
            width,
            label="Saved scenario",
            color="#086548",
        )
        comparison_ax.set_xticks(positions)
        comparison_ax.set_xticklabels(scenario_df["Name"], rotation=25)
        comparison_ax.set_ylabel("kWh/month")
        comparison_ax.legend(frameon=False)
        comparison_ax.grid(axis="y", alpha=0.2)
        comparison_ax.spines[["top", "right"]].set_visible(False)
        comparison_chart.tight_layout()
        st.pyplot(comparison_chart, width="stretch")
        plt.close(comparison_chart)


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    '<div class="app-footer">Smart Electricity Consumption &amp; '
    'Optimisation | Use electricity wisely. Predictions and savings are '
    'estimates based on the details you enter.</div>',
    unsafe_allow_html=True,
)
