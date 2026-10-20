
import streamlit as st
import PyPDF2
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="HireSmart AI Tool",
    layout="wide",
    page_icon="🤖"
)

# ==================================================
# PROFESSIONAL UI DESIGN - CSS ONLY
# ==================================================

st.markdown("""
<style>
/* Main application */
.stApp {
    background-color: #F5F7FB;
    color: #172B4D;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Main headings */
h1, h2, h3 {
    color: #172B4D !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown {
    color: #475569;
}

/* Hero header */
.hero {
    background: linear-gradient(120deg, #142D4E 0%, #1E4976 100%);
    padding: 35px 38px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(20, 45, 78, 0.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 34px !important;
    margin: 0 0 10px 0;
    letter-spacing: -1px;
}

.hero p {
    color: #D8E6F5 !important;
    font-size: 15px;
    margin: 0;
    line-height: 1.7;
}

.hero-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.12);
    color: #D8E6F5;
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin-bottom: 14px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #17365D !important;
}

section[data-testid="stSidebar"] p {
    color: #475569;
    font-size: 13px;
    line-height: 1.8;
}

/* Text areas */
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #D5DFEB !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #377DCE !important;
    box-shadow: 0 0 0 2px rgba(55, 125, 206, 0.12) !important;
}

/* Labels */
.stTextArea label,
.stFileUploader label {
    color: #172B4D !important;
    font-weight: 600 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px dashed #A8BDD5;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stFileUploader"] section {
    background-color: #F8FAFD;
    border-radius: 9px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    min-height: 44px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"] {
    background: #2463A6;
    color: #FFFFFF;
    border: 1px solid #2463A6;
}

.stButton > button[kind="primary"]:hover {
    background: #194C85;
    border-color: #194C85;
    box-shadow: 0 4px 12px rgba(36, 99, 166, 0.2);
}

.stDownloadButton > button {
    background: #FFFFFF;
    color: #2463A6;
    border: 1px solid #2463A6;
}

.stDownloadButton > button:hover {
    background: #EDF5FF;
    color: #194C85;
    border-color: #194C85;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 22px;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 13px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #17365D !important;
    font-size: 30px;
    font-weight: 700;
}

/* Data table */
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}

/* Candidate detail expanders */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E0E7F0;
    border-radius: 11px;
    margin-bottom: 10px;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: #17365D;
    font-weight: 600;
    padding: 12px;
}

[data-testid="stExpander"] summary:hover {
    color: #2463A6;
}

/* Alerts and messages */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Section divider */
.section-label {
    color: #64748B;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 12px;
    margin-bottom: 8px;
}

/* Responsive layout */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .hero {
        padding: 24px 20px;
    }

    .hero h1 {
        font-size: 26px !important;
    }

    [data-testid="stMetric"] {
        padding: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# ORIGINAL FUNCTIONS - UNCHANGED
# ==================================================

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            pt = page.extract_text()
            if pt:
                text += pt + "\n"
        return text
    except Exception:
        return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_matched_keywords(jd_text, cv_text, top_n=15):
    jd_words = set(clean_text(jd_text).split())
    cv_words = set(clean_text(cv_text).split())
    matched = [w for w in jd_words.intersection(cv_words) if len(w) > 2]
    return matched[:top_n]


def generate_reasoning(score, matched_keywords):
    if score >= 70:
        return f"Highly relevant. Strong overlap on {len(matched_keywords)} key skills: {', '.join(matched_keywords[:5])}."
    elif score >= 45:
        return f"Moderately relevant. Matches on {len(matched_keywords)} keywords: {', '.join(matched_keywords[:5])}. Some JD requirements missing."
    else:
        return f"Low relevance. Only {len(matched_keywords)} keywords matched. Major JD skills not found."


# ==================================================
# PROFESSIONAL HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI-POWERED RECRUITMENT PLATFORM</div>
    <h1>🤖 HireSmart</h1>
    <p>
        Intelligent Resume Screening & Candidate Ranking<br>
        Upload CVs, compare qualifications with job requirements,
        and generate an organized screening report.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR - ORIGINAL ARCHITECTURE INFORMATION
# ==================================================

with st.sidebar:
    st.markdown("## 🧠 Project Architecture")
    st.caption("Technical overview for your project demonstration.")
    st.divider()

    st.markdown("""
**01 · Python & Data Processing**

Python, Pandas and NumPy for data handling and score calculations.

**02 · Natural Language Processing**

The `clean_text()` function processes text and removes punctuation.

**03 · TF-IDF & Similarity**

`TfidfVectorizer` and `cosine_similarity` compare the JD with each CV.

**04 · Keyword Retrieval**

Common words between the JD and CV are retrieved as matching evidence.

**05 · Screening Workflow**

- Parser: `extract_text_from_pdf()`
- Ranker: TF-IDF similarity calculation
- Reasoner: `generate_reasoning()`

**06 · Results & Reporting**

Candidate ranking, screening summaries and CSV export.
""")

    st.divider()
    st.caption("HireSmart | Resume Screening Project")


# ==================================================
# ORIGINAL INPUTS
# ==================================================

st.markdown(
    '<div class="section-label">01 — Screening Setup</div>',
    unsafe_allow_html=True
)

st.markdown("### Define your hiring requirements")
st.caption("Start by providing the job description and candidate CVs.")

jd_input = st.text_area(
    "Paste Job Description (JD) Here:",
    height=200,
    placeholder="e.g., Looking for Python developer with Pandas, Machine Learning..."
)

uploaded_files = st.file_uploader(
    "Upload CVs (Multiple PDFs Allowed)",
    type=["pdf"],
    accept_multiple_files=True
)

st.markdown("")


# ==================================================
# ORIGINAL SCREENING LOGIC - UNCHANGED
# ==================================================

if st.button("🚀 Run AI Screening", type="primary"):
    if not jd_input.strip():
        st.warning("Please paste a Job Description first.")
    elif not uploaded_files:
        st.warning("Please upload at least one CV.")
    else:
        with st.spinner("AI Agents are screening candidates..."):
            cv_texts = []
            file_names = []
            raw_texts = []

            for pdf in uploaded_files:
                text = extract_text_from_pdf(pdf)
                cv_texts.append(clean_text(text))
                raw_texts.append(text)
                file_names.append(pdf.name)

            all_docs = [clean_text(jd_input)] + cv_texts

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(all_docs)

            jd_vector = tfidf_matrix[0:1]
            cv_vectors = tfidf_matrix[1:]

            scores = cosine_similarity(
                jd_vector, cv_vectors
            ).flatten()

            results = []

            for i in range(len(file_names)):
                score_percent = round(float(scores[i]) * 100, 2)

                matched = get_matched_keywords(
                    jd_input, cv_texts[i]
                )

                reasoning = generate_reasoning(
                    score_percent, matched
                )

                if score_percent >= 60:
                    decision = "✅ Shortlist"
                elif score_percent >= 35:
                    decision = "⚠️ Maybe"
                else:
                    decision = "❌ Reject"

                results.append({
                    "File Name": file_names[i],
                    "Match Score %": score_percent,
                    "Decision": decision,
                    "Matched Keywords": ", ".join(matched),
                    "AI Reasoning": reasoning,
                    "Raw CV": raw_texts[i]
                })

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                by="Match Score %",
                ascending=False
            ).reset_index(drop=True)

            st.success(
                f"Screening Complete! {len(result_df)} candidates analyzed."
            )

            # Results dashboard
            st.markdown(
                '<div class="section-label">02 — Screening Overview</div>',
                unsafe_allow_html=True
            )
            st.markdown("### Candidate Screening Dashboard")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total CVs",
                len(result_df)
            )

            col2.metric(
                "Shortlisted",
                len(
                    result_df[
                        result_df['Decision'].str.contains("Shortlist")
                    ]
                )
            )

            col3.metric(
                "Avg Score",
                f"{result_df['Match Score %'].mean():.1f}%"
            )

            # Candidate ranking table
            st.markdown("")
            st.markdown("### 📊 Candidate Rankings")
            st.caption(
                "Candidates are displayed in descending order of similarity."
            )

            display_df = result_df[
                [
                    "File Name",
                    "Match Score %",
                    "Decision",
                    "AI Reasoning"
                ]
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

            # Detailed candidate analysis
            st.markdown("")
            st.markdown(
                '<div class="section-label">03 — Individual Reports</div>',
                unsafe_allow_html=True
            )
            st.markdown("### 🔍 Detailed Candidate Analysis")

            for idx, row in result_df.iterrows():
                with st.expander(
                    f"{row['Decision']} - {row['File Name']} - "
                    f"{row['Match Score %']}%"
                ):
                    st.markdown("**Screening Reasoning**")
                    st.write(row['AI Reasoning'])

                    st.markdown("**Retrieved Keyword Evidence**")
                    st.write(row['Matched Keywords'])

                    st.markdown("**CV Text Preview**")
                    st.text_area(
                        "CV Preview (first 1000 chars)",
                        row['Raw CV'][:1000],
                        height=150,
                        key=f"preview_{idx}"
                    )

            # CSV download
            st.markdown("")
            st.markdown("### 📥 Export Screening Report")

            csv = result_df.drop(
                columns=["Raw CV"]
            ).to_csv(index=False).encode('utf-8')

            st.download_button(
                "📥 Download Screening Report (CSV)",
                csv,
                "hiring_report.csv",
                "text/csv"
            )

else:
    st.info(
        "Upload CVs and enter JD, then click Run AI Screening "
        "to see Agentic AI results."
    )
