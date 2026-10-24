"""Streamlit app for resistor detection and color-code calculations.

Run with:
    .\\venv\\Scripts\\python.exe -m streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "resistor_detector_fast"
    / "weights"
    / "best.pt"
)

COLOR_VALUES = {
    "Black": 0,
    "Brown": 1,
    "Red": 2,
    "Orange": 3,
    "Yellow": 4,
    "Green": 5,
    "Blue": 6,
    "Violet": 7,
    "Gray": 8,
    "White": 9,
}
COLOR_HEX = {
    "Black": "#27272a",
    "Brown": "#8b4e2f",
    "Red": "#e5484d",
    "Orange": "#ed8b36",
    "Yellow": "#f2cf43",
    "Green": "#46a878",
    "Blue": "#438bd0",
    "Violet": "#9671c8",
    "Gray": "#9ca3af",
    "White": "#f8fafc",
    "Gold": "#d7ad43",
    "Silver": "#b9c4d0",
    "None": "#f2f5f9",
}
MULTIPLIERS = {
    "Black": 1,
    "Brown": 10,
    "Red": 100,
    "Orange": 1_000,
    "Yellow": 10_000,
    "Green": 100_000,
    "Blue": 1_000_000,
    "Violet": 10_000_000,
    "Gray": 100_000_000,
    "White": 1_000_000_000,
    "Gold": 0.1,
    "Silver": 0.01,
}
TOLERANCES = {
    "Brown": 1,
    "Red": 2,
    "Green": 0.5,
    "Blue": 0.25,
    "Violet": 0.1,
    "Gray": 0.05,
    "Gold": 5,
    "Silver": 10,
    "None": 20,
}
DIGIT_COLORS = list(COLOR_VALUES)
MULTIPLIER_COLORS = DIGIT_COLORS + ["Gold", "Silver"]
TOLERANCE_COLORS = [
    "Brown",
    "Red",
    "Green",
    "Blue",
    "Violet",
    "Gray",
    "Gold",
    "Silver",
    "None",
]

st.set_page_config(
    page_title="RESIFINDER · AI & Calculator",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #edf2f9;
        --muted: #a3afc2;
        --red: #ff5b64;
        --red-dark: #ff737a;
        --blue: #a8ddff;
        --blue-strong: #65bded;
        --line: #293649;
        --paper: #0b111b;
        color-scheme: dark;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 50% -25%, rgba(40, 79, 113, .25), transparent 55%),
            var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(11, 17, 27, .82); }
    .block-container {
        max-width: 1380px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }
    .hero {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2.5rem;
        min-height: 0;
        padding: 2.8rem 3.2rem;
        border: 1px solid #2b3b50;
        border-radius: 28px;
        background:
            radial-gradient(ellipse at 88% 8%, rgba(255, 91, 100, .12), transparent 32%),
            radial-gradient(ellipse at 8% 100%, rgba(56, 121, 169, .14), transparent 44%),
            linear-gradient(125deg, #121d2b 0%, #111b29 60%, #172439 100%);
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 55px rgba(0, 0, 0, .24), inset 0 1px rgba(255,255,255,.045);
    }
    .hero-copy { position: relative; z-index: 1; max-width: 800px; }
    .brand-row {
        display: flex;
        align-items: center;
        gap: .9rem;
        margin-bottom: 1.5rem;
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.8rem, 6vw, 5rem);
        font-weight: 700;
        letter-spacing: -.085em;
        line-height: .98;
    }
    .brand-mark {
        display: grid;
        flex: 0 0 auto;
        width: 58px;
        height: 58px;
        place-items: center;
        border-radius: 18px;
        background: var(--red);
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        letter-spacing: 0;
        box-shadow: 0 8px 28px rgba(255, 91, 100, .25);
    }
    .eyebrow {
        color: var(--blue);
        font-size: .7rem;
        font-weight: 700;
        letter-spacing: .19em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: #f4f7fc;
        font-size: clamp(1.5rem, 2.7vw, 2.3rem);
        font-weight: 500;
        letter-spacing: -.04em;
        line-height: 1.15;
        margin: .65rem 0 .55rem;
    }
    .hero h1 span { color: var(--red); }
    .hero-points {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.15rem;
    }
    .hero-point {
        padding: .42rem .75rem;
        border: 1px solid rgba(168, 221, 255, .16);
        border-radius: 999px;
        background: rgba(168, 221, 255, .055);
        color: #c9d6e5;
        font-size: .78rem;
        font-weight: 600;
    }
    .hero p {
        max-width: 570px;
        color: #aebbd0;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
    }
    div[data-testid="stTabs"] > div:first-child {
        gap: .65rem;
        border-bottom: 0;
        margin: 0 0 1.2rem;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        min-height: 50px;
        padding: .65rem 1.15rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: #111a27;
        color: var(--muted);
        font-weight: 600;
        transition: background .18s ease, border-color .18s ease, transform .18s ease;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        border-color: #536880;
        color: var(--ink);
        transform: translateY(-1px);
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--red-dark);
        border-color: rgba(255, 91, 100, .52);
        background: linear-gradient(135deg, rgba(255, 91, 100, .14), rgba(255, 91, 100, .055));
        box-shadow: inset 0 0 0 1px rgba(255, 91, 100, .08);
    }
    .section-heading {
        display: flex;
        align-items: baseline;
        gap: .7rem;
        margin: .25rem 0 .3rem;
    }
    .section-heading h2 {
        margin: 0;
        font-size: 1.35rem;
        letter-spacing: -.035em;
    }
    .section-caption {
        color: var(--muted);
        font-size: .9rem;
        margin: 0 0 1rem;
    }
    .step-badge {
        display: inline-grid;
        width: 25px;
        height: 25px;
        place-items: center;
        border-radius: 50%;
        background: rgba(255, 91, 100, .15);
        color: var(--red-dark);
        font-size: .72rem;
        font-weight: 700;
    }
    .soft-card {
        padding: 1.35rem 1.5rem;
        border: 1px solid #2b3b50;
        border-radius: 20px;
        background: linear-gradient(145deg, #172333, #121d2a);
        box-shadow: 0 12px 32px rgba(0, 0, 0, .2);
    }
    .result-card {
        min-height: 220px;
        padding: 1.6rem;
        border: 1px solid rgba(101, 189, 237, .28);
        border-radius: 20px;
        background:
            radial-gradient(ellipse at 100% 0%, rgba(101, 189, 237, .1), transparent 44%),
            linear-gradient(145deg, #172536 5%, #132638 100%);
        box-shadow: 0 12px 32px rgba(0, 0, 0, .2);
    }
    .result-label {
        color: var(--muted);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .result-value {
        margin: .45rem 0 .25rem;
        color: var(--red-dark);
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 700;
        letter-spacing: -.055em;
    }
    .range-value { color: #bac8d9; font-size: .9rem; line-height: 1.7; }
    .resistor-visual {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100px;
        margin: 1rem 0;
    }
    .resistor-lead {
        width: 13%;
        height: 6px;
        background: #7f96a9;
    }
    .resistor-body {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        width: 70%;
        max-width: 340px;
        height: 62px;
        padding: 0 24px;
        border: 1px solid #4e4237;
        border-radius: 22px;
        background: linear-gradient(180deg, #e8c99f, #c79c6c 55%, #98704f);
        box-shadow: inset 0 5px 7px rgba(255,255,255,.35), 0 5px 10px rgba(0,0,0,.28);
    }
    .resistor-band {
        width: 13px;
        height: 60px;
        border: 1px solid rgba(30, 41, 59, .18);
        border-radius: 2px;
    }
    [data-testid="stFileUploader"] {
        padding: .85rem;
        border: 1px dashed #526b84;
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(20, 32, 47, .92), rgba(14, 23, 35, .92));
        transition: border-color .18s ease, background .18s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--blue-strong);
        background: linear-gradient(145deg, rgba(24, 39, 57, .96), rgba(15, 26, 40, .96));
    }
    [data-testid="stFileUploader"] section { border: 0; background: transparent; }
    [data-testid="stFileUploader"] button {
        border: 1px solid #61343b;
        border-radius: 11px;
        background: linear-gradient(135deg, #3a222b, #291b24);
        color: #ff9499;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: var(--red);
        color: #ffb1b4;
    }
    [data-testid="stMetric"] {
        padding: 1.05rem 1.2rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: #151e2b;
        box-shadow: 0 5px 24px rgba(0, 0, 0, .18);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .8rem; font-weight: 600; }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
    }
    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, .24);
    }
    [data-testid="stDataFrame"] { overflow: hidden; border: 1px solid var(--line); border-radius: 14px; }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); background: #111a27; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input {
        border-color: #33445a;
        background: #111a27;
        color: var(--ink);
    }
    [data-testid="stRadio"] label,
    [data-testid="stSlider"] label { color: var(--ink); }
    [data-testid="stRadio"] [role="radiogroup"] {
        gap: .55rem;
    }
    [data-testid="stRadio"] [role="radiogroup"] label {
        padding: .5rem .8rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #111a27;
    }
    [data-testid="stExpander"] {
        border-color: var(--line);
        border-radius: 14px;
        background: #111a27;
    }
    .footer-note {
        margin-top: 2.5rem;
        padding: 1.1rem 0;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: .78rem;
        text-align: center;
    }
    @media (max-width: 760px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { min-height: 0; padding: 1.65rem; border-radius: 22px; }
        .hero { flex-direction: column; align-items: stretch; }
        .brand-row { gap: .65rem; margin-bottom: 1rem; font-size: clamp(2.5rem, 11vw, 3.7rem); }
        .brand-mark { width: 48px; height: 48px; border-radius: 14px; font-size: 1.45rem; }
        .hero p { font-size: .98rem; }
        div[data-testid="stTabs"] > div:first-child { gap: .4rem; }
        div[data-testid="stTabs"] button[role="tab"] { min-height: 44px; padding: .45rem .65rem; font-size: .82rem; }
        .resistor-visual { height: 82px; }
        .resistor-body { gap: 9px; padding: 0 15px; }
        .resistor-band { width: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained resistor model...")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def format_resistance(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:g} GΩ"
    if value >= 1_000_000:
        return f"{value / 1_000_000:g} MΩ"
    if value >= 1_000:
        return f"{value / 1_000:g} kΩ"
    return f"{value:g} Ω"


def render_color_calculator() -> None:
    st.markdown(
        """
        <div class="section-heading"><span class="step-badge">01</span><h2>Build your color code</h2></div>
        <p class="section-caption">Choose the bands from left to right. Your resistance updates as you go.</p>
        """,
        unsafe_allow_html=True,
    )

    band_count = st.radio(
        "Resistor type",
        options=[4, 5],
        format_func=lambda value: f"{value}-band resistor",
        horizontal=True,
        key="band_count",
    )
    digit_count = band_count - 2
    default_digits = (
        ["Red", "Violet"] if band_count == 4 else ["Brown", "Black", "Black"]
    )

    with st.container(border=True):
        st.markdown("**Select each band color**")
        columns = st.columns(band_count, gap="small")
        selected_colors = []
        for index, column in enumerate(columns):
            if index < digit_count:
                options = DIGIT_COLORS
                label = f"Band {index + 1} · digit"
                default = default_digits[index]
            elif index == digit_count:
                options = MULTIPLIER_COLORS
                label = f"Band {index + 1} · multiplier"
                default = "Yellow" if band_count == 4 else "Red"
            else:
                options = TOLERANCE_COLORS
                label = f"Band {index + 1} · tolerance"
                default = "Gold" if band_count == 4 else "Brown"

            with column:
                color = st.selectbox(
                    label,
                    options,
                    index=options.index(default),
                    key=f"band_{band_count}_{index}",
                    help="Select the color painted on this resistor band.",
                )
                selected_colors.append(color)

        digit_value = 0
        for color in selected_colors[:digit_count]:
            digit_value = digit_value * 10 + COLOR_VALUES[color]
        resistance = digit_value * MULTIPLIERS[selected_colors[digit_count]]
        tolerance = TOLERANCES[selected_colors[-1]]
        lower_bound = resistance * (1 - tolerance / 100)
        upper_bound = resistance * (1 + tolerance / 100)

    result_column, visual_column = st.columns([1, 1], gap="large")
    with result_column:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Calculated resistance</div>
                <div class="result-value">{format_resistance(resistance)}</div>
                <div class="range-value">Tolerance: ±{tolerance:g}%</div>
                <div class="range-value">Range: {format_resistance(lower_bound)} – {format_resistance(upper_bound)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with visual_column:
        bands_html = "".join(
            f'<span class="resistor-band" style="background:{COLOR_HEX[color]}"></span>'
            for color in selected_colors
        )
        st.markdown(
            f"""
            <div class="soft-card">
                <div class="result-label">Your band sequence</div>
                <div class="resistor-visual">
                    <span class="resistor-lead"></span>
                    <span class="resistor-body">{bands_html}</span>
                    <span class="resistor-lead"></span>
                </div>
                <div style="text-align:center;color:#a3afc2;font-size:.9rem">
                    {' · '.join(selected_colors)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("How to read resistor bands"):
        st.markdown(
            "- **4-band:** first two bands are digits, third is the multiplier, fourth is tolerance.\n"
            "- **5-band:** first three bands are digits, fourth is the multiplier, fifth is tolerance.\n"
            "- Read from the end where the bands are closest together; the tolerance band is usually spaced apart."
        )


def render_detector(confidence: float, model_path_text: str) -> None:
    st.markdown(
        """
        <div class="section-heading"><span class="step-badge">1</span><h2>Start with a photo</h2></div>
        <p class="section-caption">Choose a clear, well-lit image. We’ll mark detected resistors and show the confidence for each match.</p>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Add a resistor photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Choose a JPG, PNG, or WebP image. A close, in-focus photo works best.",
        key="detector_upload",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="soft-card" style="text-align:center;margin-top:1rem">
                <div style="font-size:2rem;margin-bottom:.45rem">📷</div>
                <strong>Your results will show up here</strong>
                <div class="section-caption" style="margin:.4rem 0 0">Tip: place one resistor against a plain background and avoid glare.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        model_path = Path(model_path_text).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_DIR / model_path
        model_path = model_path.resolve()

        if not model_path.is_file():
            st.error(f"Model checkpoint not found: {model_path}")
            st.info("Check that the trained best.pt checkpoint is included in the project.")
            return

        image = Image.open(uploaded_file).convert("RGB")
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not open the image or model path: {error}")
        return

    try:
        with st.spinner("Looking for resistor values..."):
            model = load_model(str(model_path))
            predictions = model.predict(
                source=image,
                imgsz=320,
                conf=confidence,
                device="cpu",
                verbose=False,
            )
    except Exception as error:
        st.error(f"Detection failed: {error}")
        return

    result = predictions[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "Resistor": model.names[class_id],
                    "Confidence": float(box.conf.item()),
                }
            )

    detections.sort(key=lambda item: item["Confidence"], reverse=True)
    annotated_image = Image.fromarray(result.plot()[..., ::-1])
    image_column, result_column = st.columns([1.1, 0.9], gap="large")

    with image_column:
        st.markdown(
            '<div class="section-heading"><span class="step-badge">2</span><h2>Detection preview</h2></div>',
            unsafe_allow_html=True,
        )
        st.image(annotated_image, use_container_width=True)

    with result_column:
        st.markdown(
            '<div class="section-heading"><span class="step-badge">3</span><h2>Results</h2></div>',
            unsafe_allow_html=True,
        )
        st.metric("Resistors detected", len(detections))
        if detections:
            st.dataframe(
                [
                    {
                        "Resistor": item["Resistor"],
                        "Confidence": f"{item['Confidence']:.1%}",
                    }
                    for item in detections
                ],
                hide_index=True,
                use_container_width=True,
            )
            best_prediction = detections[0]
            st.markdown("**Best match**")
            st.markdown(f"### {best_prediction['Resistor']}")
            st.write(f"Confidence: {best_prediction['Confidence']:.1%}")

            if best_prediction["Confidence"] < 0.50:
                st.warning("Low confidence — check the resistor value independently.")
            if len(detections) > 1:
                st.info("Multiple detections found. Check the preview to verify each box.")
        else:
            st.info("No resistors detected. Try a clearer photo or lower the confidence setting.")

        st.caption(f"Model: {model_path.name} · Minimum confidence: {confidence:.0%}")


st.markdown(
    """
    <div class="hero">
        <div class="hero-copy">
            <div class="brand-row"><span class="brand-mark">Ω</span> RESIFINDER</div>
            <div class="eyebrow">Your friendly resistor workbench</div>
            <h1>Find the value.<br><span>Get back to building.</span></h1>
            <p>Upload a photo for an AI-assisted check, or enter the color bands to calculate resistance. Pick a tool below to get started.</p>
            <div class="hero-points">
                <span class="hero-point">✦ AI photo detection</span>
                <span class="hero-point">◎ 4- and 5-band calculator</span>
                <span class="hero-point">↗ Clear tolerance range</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Detection settings")
    st.caption("These settings only affect AI image detection.")
    confidence = st.slider(
        "Minimum confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.10,
        step=0.05,
        help="Detections below this confidence are hidden.",
    )
    model_path_text = st.text_input(
        "Model checkpoint",
        value=str(DEFAULT_MODEL),
        help="Path to the trained best.pt model checkpoint.",
    )
    st.caption("The color-code calculator works independently of the AI model.")

detector_tab, calculator_tab = st.tabs(
    ["🔎  AI resistor detector", "🎨  Color-code calculator"]
)

with detector_tab:
    render_detector(confidence, model_path_text)

with calculator_tab:
    render_color_calculator()

st.markdown(
    '<div class="footer-note">RESIFINDER &nbsp;·&nbsp; Built for makers, students, and curious minds.</div>',
    unsafe_allow_html=True,
)
