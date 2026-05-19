CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"], button, input, select, textarea {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Chrome ───────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

.stApp { background-color: #FFFFFF !important; }
.main .block-container {
    padding-top: 2.5rem !important;
    padding-left: 3.5rem !important;
    padding-right: 3.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 100% !important;
}

/* ── Hero header ──────────────────────────────── */
.rf-hero {
    padding-bottom: 2rem;
    border-bottom: 2px solid #0D9488;
    margin-bottom: 2rem;
}
.rf-logo {
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.05em;
    color: #0F172A;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.rf-logo em { color: #0D9488; font-style: normal; }
.rf-desc {
    font-size: 0.9rem;
    color: #64748B;
    font-weight: 400;
    line-height: 1.6;
    max-width: 560px;
}

/* ── Search bar ───────────────────────────────── */
[data-testid="stTextInput"] input {
    border-radius: 999px !important;
    border: 2px solid #E2E8F0 !important;
    padding: 10px 20px !important;
    font-size: 0.9rem !important;
    background: #F8FAFC !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #0D9488 !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.12) !important;
}
[data-testid="stTextInput"] label { display: none !important; }

/* ── Filter labels ────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label {
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #94A3B8 !important;
}

/* ── Filter inputs ────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
    background: #F8FAFC !important;
    font-size: 0.84rem !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stMultiSelect"] > div > div:focus-within {
    border-color: #0D9488 !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.10) !important;
}

/* ── Multiselect tags ─────────────────────────── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #F0FDFA !important;
    color: #0F766E !important;
    border-radius: 8px !important;
}

/* ── Slider ───────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background-color: #0D9488 !important;
    border-color: #0D9488 !important;
}

/* ── Reset button ─────────────────────────────── */
[data-testid="baseButton-secondary"] {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
    background: #F8FAFC !important;
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #0D9488 !important;
    color: #0D9488 !important;
    background: #F0FDFA !important;
}

/* ── Divider ──────────────────────────────────── */
.rf-divider {
    height: 1px;
    background: #E2E8F0;
    margin: 1.2rem 0 1.5rem;
}

/* ── Stats bar ────────────────────────────────── */
.rf-stats {
    display: flex;
    gap: 36px;
    align-items: baseline;
    padding: 1rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.2rem;
}
.rf-stat-n {
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #0F172A;
}
.rf-stat-l {
    font-size: 0.6rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94A3B8;
    margin-left: 6px;
}
.rf-caption {
    font-size: 0.73rem;
    color: #94A3B8;
    margin-bottom: 0.6rem;
}

/* ── Table ────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #E2E8F0 !important;
}

/* ── Rec section ──────────────────────────────── */
.rf-rec-name {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0F172A;
    margin-bottom: 2px;
}
.rf-rec-name em { color: #0D9488; font-style: normal; }
.rf-rec-sub {
    font-size: 0.68rem;
    color: #94A3B8;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── Detail card ──────────────────────────────── */
.rf-card {
    background: #F8FAFC;
    border-radius: 12px;
    padding: 22px 24px;
    border: 1px solid #E2E8F0;
}
.rf-card table { width: 100%; border-collapse: collapse; }
.rf-card td { padding: 6px 0; vertical-align: top; }
.rf-card td.lbl {
    font-size: 0.59rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94A3B8;
    width: 36%;
    padding-top: 8px;
}
.rf-card td.val {
    font-size: 0.86rem;
    font-weight: 500;
    color: #0F172A;
}

hr { border-color: #E2E8F0 !important; }
</style>
"""
