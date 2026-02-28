import streamlit as st
from utils.pdf_parser import extract_text_from_pdf
from utils.similarity import (
    calculate_similarity, 
    analyze_keyword_gap, 
    get_top_keywords,
    extract_mandatory_skills,
    calculate_category_scores,
    calculate_mandatory_coverage,
    calculate_hybrid_score,
    generate_decision_and_explanation
)
from visualization.charts import plot_match_scores

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")

# ===== HERO SECTION =====
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg,#1f2937 0%,#4a5568 100%);
        padding:40px;
        border-radius:10px;
        text-align:center;
        color:#f7fafc;
        margin-bottom:30px;
    ">
      <h1 style="margin:0;font-size:3rem;font-weight:700;">Resume Screening System</h1>
      <p style="margin:8px 0 0;font-size:1.1rem;color:#e2e8f0;">
         AI‑Powered Hybrid Scoring · Explainability · Skill Gap Analysis
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# KeyFeatures highlight
st.markdown("""
<div style="
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-bottom: 30px;
    flex-wrap: wrap;
">
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 24px;">🤖</p>
        <p style="margin: 8px 0 0 0; color: #2d3748; font-weight: 600; font-size: 13px;">HYBRID SCORING</p>
    </div>
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 24px;">📊</p>
        <p style="margin: 8px 0 0 0; color: #2d3748; font-weight: 600; font-size: 13px;">SKILL ANALYSIS</p>
    </div>
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 24px;">✨</p>
        <p style="margin: 8px 0 0 0; color: #2d3748; font-weight: 600; font-size: 13px;">EXPLAINABILITY</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===== INPUT SECTION =====
st.markdown("<h2 style='color: #7C3AED; font-weight:700; font-size:1.75rem; margin-top: 30px;'>📋 Screen Your Resume</h2>", unsafe_allow_html=True)

# ===== STYLED UPLOADER =====
st.markdown("""
<div style="
    background-color: #f7fafc;
    border: 2px dashed #cbd5e0;
    border-radius: 8px;
    padding: 1px;
">
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ===== JOB DESCRIPTION INPUT =====
st.markdown("<p style='color: #2d3748; font-weight: 600; margin: 20px 0 8px 0;'>Paste Job Description:</p>", unsafe_allow_html=True)
job_description = st.text_area(
    "Paste the job description here",
    height=160,
    label_visibility="collapsed",
    placeholder="Paste the complete job description here. Include required skills, responsibilities, and qualifications..."
)

# copy inputs to final vars (avoids undefined-variable later)
uploaded_files_final = uploaded_files
job_description_final = job_description

# ===== STYLED ANALYZE BUTTON =====
col1, col2, col3 = st.columns([1, 1.5, 1])
with col2:
    button_clicked = st.button(
        "🚀 Analyze Match",
        use_container_width=True,
        key="analyze_button"
    )

# Custom button styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
    color: white;
    font-weight: 600;
    font-size: 16px;
    padding: 12px 24px !important;
    border: none;
    border-radius: 6px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

div.stButton > button:first-child:hover {
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR (UPDATED) ----------------
with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:
    - Assess how well your resume matches a job description.
    - Extract mandatory skills ("must have", "required", etc)
    - Calculate skill category match scores.
    - Get hybrid final score + decision.
    - View keyword analysis & gap assessment.
    """)
    
    st.header("How it works")
    st.write("""
    1. Upload your resume in PDF format.
    2. Paste the job description.
    3. Click **Analyze Match**
    4. Review hybrid score, decision & explainability panel
    """)
    
    st.header("Scoring Breakdown")
    st.write("""
    **Final Score = **
    - 50% Semantic Similarity
    - 30% Skill Category Match
    - 20% Mandatory Skill Coverage
    
    **Decision:**
    - ≥ 75 → **Shortlist**
    - 50-74 → **Review**
    - < 50 → **Reject**
    """)

# ---------------- MAIN LOGIC ----------------

# ----- view helpers -----
def render_recruiter_view(name, analysis):
    """Render full detailed report for recruiters inside its own card container."""
    with st.container(border=True):
        st.markdown(f"### 📄 {name}")
        
        hybrid_score = analysis['hybrid_score']
        decision = analysis['decision']
        explanation = analysis['explanation']
        
        # decision card
        if decision == 'Shortlist':
            st.success(f"✅ **{decision}** — Score: **{hybrid_score}%**")
        elif decision == 'Review':
            st.info(f"🔄 **{decision}** — Score: **{hybrid_score}%**")
        else:
            st.error(f"❌ **{decision}** — Score: **{hybrid_score}%**")
        
        st.caption(explanation)
        
        # Score breakdown
        st.markdown("**📊 Score Breakdown:**")
        pb_col1, pb_col2, pb_col3 = st.columns(3)
        with pb_col1:
            semantic_val = analysis['semantic_score'] / 100
            st.metric("Semantic Similarity", f"{analysis['semantic_score']}%")
            st.progress(semantic_val)
        with pb_col2:
            category_val = analysis['category_avg'] / 100
            st.metric("Skill Category Match", f"{analysis['category_avg']}%")
            st.progress(category_val)
        with pb_col3:
            mandatory_val = analysis['mandatory_coverage'] / 100
            st.metric("Mandatory Coverage", f"{analysis['mandatory_coverage']}%")
            st.progress(mandatory_val)
        
        st.divider()
        
        # missing skills badges
        st.markdown("**⚠️ Top 5 Skills to Add:**")
        gap = analysis['gap_analysis']
        missing_keywords = gap['missing_keywords']
        if missing_keywords:
            top_5_missing = [kw[0] for kw in missing_keywords[:5]]
            badges_html = "".join([f"<span style='display:inline-block;background-color:#ffe5e5;color:#c41e3a;padding:8px 14px;margin:5px 5px 5px 0;border-radius:20px;font-weight:500;font-size:13px;border:1px solid #ffcccc;'>{skill}</span>" for skill in top_5_missing])
            st.markdown(badges_html, unsafe_allow_html=True)
        else:
            st.success("✅ All required skills covered!")
        
        st.divider()
        
        # category breakdown
        st.markdown("**📊 Skill Category Breakdown:**")
        category_scores = analysis['category_scores'].copy()
        category_scores.pop('average')
        for category, score in category_scores.items():
            st.write(f"• {category.replace('_',' ').title()}: **{score}%**")
        
        st.divider()
        
        # mandatory summary
        st.markdown("**🔑 Mandatory Skills Coverage:**")
        mandatory_skills = analysis['mandatory_skills']
        resume_text_lower = analysis['resume_text'].lower()
        found = len([s for s in mandatory_skills if s in resume_text_lower])
        st.write(f"• Found: **{found}/{len(mandatory_skills)}** ({analysis['mandatory_coverage']}%)")
        
        st.divider()
        
        # matching keywords
        st.markdown("**✅ Matching Keywords:**")
        matching = [kw[0] for kw in gap['matching_keywords'][:10]]
        if matching:
            st.write(", ".join(matching))
            if len(gap['matching_keywords']) > 10:
                st.caption(f"... and {len(gap['matching_keywords'])-10} more")
        else:
            st.caption("None found")




if button_clicked:

    if not uploaded_files_final:
        st.warning("Please upload at least one resume PDF.")
        st.stop()

    if not job_description_final:
        st.warning("Please enter a job description.")
        st.stop()

    results = []
    detailed_analysis = {}

    with st.spinner("Analyzing resumes..."):
        for file in uploaded_files_final:
            data = file.read()
            resume_text = extract_text_from_pdf(data)

            if resume_text:
                semantic_score = calculate_similarity(resume_text, job_description_final)
                mandatory_skills = extract_mandatory_skills(job_description_final)
                mandatory_coverage = calculate_mandatory_coverage(resume_text, mandatory_skills)
                category_scores = calculate_category_scores(resume_text, job_description_final)
                category_avg = category_scores['average']
                hybrid_score = calculate_hybrid_score(semantic_score, category_avg, mandatory_coverage)
                decision_data = generate_decision_and_explanation(
                    hybrid_score, mandatory_coverage, semantic_score, category_avg
                )
                gap_analysis = analyze_keyword_gap(resume_text, job_description_final, top_missing=20)
                top_resume_keywords = get_top_keywords(resume_text, top_n=5)
                
                # store all data
                detailed_analysis[file.name] = {
                    'resume_text': resume_text,
                    'semantic_score': semantic_score,
                    'mandatory_skills': mandatory_skills,
                    'mandatory_coverage': mandatory_coverage,
                    'category_scores': category_scores,
                    'category_avg': category_avg,
                    'hybrid_score': hybrid_score,
                    'decision': decision_data['decision'],
                    'explanation': decision_data['explanation'],
                    'gap_analysis': gap_analysis,
                    'top_resume_keywords': top_resume_keywords
                }

                results.append({
                    "Resume Name": file.name,
                    "Hybrid Score": hybrid_score,
                    "Decision": decision_data['decision'],
                    "Semantic": semantic_score,
                    "Category Avg": category_avg,
                    "Mandatory %": mandatory_coverage
                })

    if results:
        results = sorted(results, key=lambda x: x["Hybrid Score"], reverse=True)

        # ===== RESULTS SUMMARY =====
        st.subheader("📊 Results Summary (Ranked by Score)")
        
        for idx, result in enumerate(results):
            resume_name = result["Resume Name"]
            final_score = result["Hybrid Score"]
            decision = result["Decision"]
            mandatory = result["Mandatory %"]
            
            st.write(f"**📄 {resume_name}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Score", f"{final_score}%")
            
            with col2:
                if decision == 'Shortlist':
                    st.success(f"✅ {decision}")
                elif decision == 'Review':
                    st.info(f"🔄 {decision}")
                else:
                    st.error(f"❌ {decision}")
            
            with col3:
                st.metric("Mandatory Coverage", f"{mandatory}%")
            
            with col4:
                rank = idx + 1
                st.metric("Rank", f"#{rank}")
            
            if idx < len(results) - 1:
                st.divider()
        
        # ===== SCORING FORMULA =====
        with st.expander("📐 Scoring Formula Explanation"):
            st.write("""
            **Final Score = 0.5 × Semantic Similarity + 0.3 × Skill Category Match + 0.2 × Mandatory Coverage**
            
            - **Semantic Similarity** (50%): How well resume content matches job description overall
            - **Skill Category Match** (30%): Average match across technical, soft, & domain skills
            - **Mandatory Coverage** (20%): % of "must have" / "required" skills found in resume
            
            **Decision Thresholds:**
            - ≥ 75% → **Shortlist** | 50-74% → **Review** | < 50% → **Reject**
            """)
        
        st.markdown("---")
        st.subheader("📋 Detailed Reports")
        
        # ===== DETAILED REPORTS =====
        for result in results:
            name = result["Resume Name"]
            analysis = detailed_analysis[name]
            render_recruiter_view(name, analysis)
