
import streamlit as st
import PyPDF2
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="HireSmart AI Tool", layout="wide", page_icon="\U0001f916")

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

st.title("\U0001f916 HireSmart - AI Resume Screener")
st.markdown("Upload multiple CVs, paste Job Description, and let AI agents screen candidates.")

with st.sidebar:
    st.header("Architecture for Viva")
    st.markdown("""
**1. Python + Pandas/NumPy:** Data handling & stats
**2. NLP:** `clean_text()` function
**3. TF-IDF & Similarity:** `TfidfVectorizer` + `cosine_similarity`
**4. RAG:** Retrieve matched keywords from CV based on JD
**5. Agentic Workflow:**
- Parser Agent: `extract_text_from_pdf()`
- Ranker Agent: `calculate_tfidf_scores()`
- Reasoner Agent: `generate_reasoning()`
**6. Generative AI:** Reasoning generation
""")

jd_input = st.text_area("Paste Job Description (JD) Here:", height=200, placeholder="e.g., Looking for Python developer with Pandas, Machine Learning...")
uploaded_files = st.file_uploader("Upload CVs (Multiple PDFs Allowed)", type=["pdf"], accept_multiple_files=True)

if st.button("\U0001f680 Run AI Screening", type="primary"):
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
            scores = cosine_similarity(jd_vector, cv_vectors).flatten()

            results = []
            for i in range(len(file_names)):
                score_percent = round(float(scores[i]) * 100, 2)
                matched = get_matched_keywords(jd_input, cv_texts[i])
                reasoning = generate_reasoning(score_percent, matched)
                if score_percent >= 60:
                    decision = "\u2705 Shortlist"
                elif score_percent >= 35:
                    decision = "\u26a0\ufe0f Maybe"
                else:
                    decision = "\u274c Reject"
                results.append({
                    "File Name": file_names[i],
                    "Match Score %": score_percent,
                    "Decision": decision,
                    "Matched Keywords": ", ".join(matched),
                    "AI Reasoning": reasoning,
                    "Raw CV": raw_texts[i]
                })

            result_df = pd.DataFrame(results)
            result_df = result_df.sort_values(by="Match Score %", ascending=False).reset_index(drop=True)
            st.success(f"Screening Complete! {len(result_df)} candidates analyzed.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total CVs", len(result_df))
            col2.metric("Shortlisted", len(result_df[result_df['Decision'].str.contains("Shortlist")]))
            col3.metric("Avg Score", f"{result_df['Match Score %'].mean():.1f}%")

            display_df = result_df[["File Name", "Match Score %", "Decision", "AI Reasoning"]]
            st.dataframe(display_df, use_container_width=True, height=400)

            st.subheader("Detailed Candidate Analysis")
            for idx, row in result_df.iterrows():
                with st.expander(f"{row['Decision']} - {row['File Name']} - {row['Match Score %']}%"):
                    st.write(f"**Reasoning:** {row['AI Reasoning']}")
                    st.write(f"**RAG Retrieved Evidence:** {row['Matched Keywords']}")
                    st.text_area("CV Preview (first 1000 chars)", row['Raw CV'][:1000], height=150, key=f"preview_{idx}")

            csv = result_df.drop(columns=["Raw CV"]).to_csv(index=False).encode('utf-8')
            st.download_button("\U0001f4e5 Download Screening Report (CSV)", csv, "hiring_report.csv", "text/csv")
else:
    st.info("Upload CVs and enter JD, then click Run AI Screening to see Agentic AI results.")
