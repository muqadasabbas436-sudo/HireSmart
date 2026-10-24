# HireSmart - AI Powered Resume Screener using Python, NLP, TF-IDF, RAG and Agentic Workflow

##  Project Description
HireSmart is an Agentic AI based CV screening system that automates the hiring process. HR can paste a Job Description and upload bulk CVs (PDFs), and the system ranks candidates by relevance using NLP and Machine Learning techniques.

##  Problem Statement and Target Users
**Problem:** For one job position, HR receives 200-300 CVs. Manual screening is time-consuming, biased, and inefficient.
**Solution:** Automate screening with AI ranking, reasoning, and evidence.
**Target Users:** HR Managers, Recruiters, Hiring Agencies, Startups.

##  Features
- Bulk PDF CV Upload
- Job Description based Matching
- Match Score % using Cosine Similarity
- Auto Decision: Shortlist / Maybe / Reject (60/35 Threshold Logic)
- AI Reasoning & Matched Keywords Evidence (RAG-like Retrieval)
- Interactive Dashboard with 3 Metrics
- Individual Candidate Expander with Preview
- Export Screening Report as CSV

##  Technologies Used
- **Frontend:** Streamlit, Custom CSS
- **Backend:** Python
- **AI/NLP:** Scikit-learn TfidfVectorizer, Cosine Similarity, NLTK (Text Processing)
- **PDF Processing:** PyPDF2
- **Data Handling:** Pandas

##  Project Architecture and Workflow
**Architecture:** Monolithic Architecture for MVP (Frontend + Backend in single Streamlit app), In-Memory Storage.

**Workflow:**
1. **Ingestion:** Extract text from JD and CV PDFs using PyPDF2
2. **Preprocessing:** Clean text, lowercasing, stopwords removal
3. **Vectorization:** `fit_transform` using TF-IDF to convert text to numerical vectors
4. **Matching:** `cosine_similarity(jd_vector, cv_vectors)` + `flatten()` to get scores
5. **Decision & UI:** 60/35 logic -> Shortlist/Maybe/Reject, Ranking, Expander, CSV Export

##  Installation and Setup
```bash
git clone https://github.com/muqadasabbas436-sudo/HireSmart.git
cd HireSmart
pip install -r requirements.txt
