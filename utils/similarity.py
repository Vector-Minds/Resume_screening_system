from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import preprocess_text, remove_stopwords
from collections import Counter
from nltk.tokenize import word_tokenize

def calculate_similarity(resume_text: str, job_description: str) -> float:
    resume_processed = remove_stopwords(preprocess_text(resume_text))
    job_processed = remove_stopwords(preprocess_text(job_description))

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2)


# ============= LEVEL 2: KEYWORD ANALYSIS FUNCTIONS =============

def extract_keywords(text: str, top_n: int = 20) -> list:
    """
    Extract top N keywords from text using TF-IDF scores.
    
    Args:
        text: Input text to extract keywords from
        top_n: Number of top keywords to return (default: 20)
    
    Returns:
        List of tuples: [(keyword, score), ...]
    """
    processed_text = remove_stopwords(preprocess_text(text))
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    tfidf_matrix = vectorizer.fit_transform([processed_text])
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # Create keyword-score pairs and sort by score
    keyword_scores = list(zip(feature_names, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keyword_scores[:top_n]


def analyze_keyword_gap(resume_text: str, job_description: str, top_missing: int = 20) -> dict:
    """
    Perform keyword gap analysis between resume and job description.
    
    Args:
        resume_text: Resume text
        job_description: Job description text
        top_missing: Number of top missing keywords to return (default: 20)
    
    Returns:
        Dictionary with:
        - 'job_keywords': Keywords from job description
        - 'missing_keywords': Keywords in job but not in resume
        - 'matching_keywords': Keywords present in both
    """
    job_keywords_set = set([kw[0] for kw in extract_keywords(job_description, 100)])
    resume_keywords_set = set([kw[0] for kw in extract_keywords(resume_text, 100)])
    
    # Get all job keywords with scores for ordering
    job_keywords_list = extract_keywords(job_description, 100)
    job_keywords_dict = {kw: score for kw, score in job_keywords_list}
    
    missing = []
    matching = []
    
    for keyword in job_keywords_set:
        if keyword not in resume_keywords_set:
            missing.append((keyword, job_keywords_dict.get(keyword, 0)))
        else:
            matching.append((keyword, job_keywords_dict.get(keyword, 0)))
    
    # Sort by score (importance)
    missing.sort(key=lambda x: x[1], reverse=True)
    matching.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'job_keywords': job_keywords_list[:20],
        'missing_keywords': missing[:top_missing],
        'matching_keywords': matching
    }


def get_top_keywords(text: str, top_n: int = 5) -> list:
    """
    Get top N frequent keywords from text (by TF-IDF score).
    
    Args:
        text: Input text
        top_n: Number of top keywords to return (default: 5)
    
    Returns:
        List of tuples: [(keyword, score), ...]
    """
    return extract_keywords(text, top_n)


# ============= LEVEL 3: HYBRID SCREENING ENGINE =============

# Skill Categories (expanded for professional assessment)
SKILL_CATEGORIES = {
    'technical': [
        'python', 'java', 'javascript', 'cpp', 'csharp', 'golang', 'rust', 'typescript',
        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle',
        'react', 'vue', 'angular', 'nodejs', 'django', 'flask', 'fastapi',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci', 'cd', 'jenkins',
        'git', 'rest', 'graphql', 'api', 'microservices', 'distributed', 'cloud',
        'ai', 'ml', 'deep', 'nlp', 'computer', 'vision', 'tensorflow', 'pytorch',
        'data', 'analytics', 'tableau', 'powerbi', 'excel', 'spark', 'hadoop'
    ],
    'soft_skills': [
        'communication', 'leadership', 'teamwork', 'problem', 'solving', 'critical',
        'thinking', 'collaboration', 'managing', 'organize', 'planning', 'delegation',
        'presentation', 'negotiation', 'mentoring', 'coaching', 'adaptable', 'flexible'
    ],
    'domain': [
        'finance', 'banking', 'healthcare', 'ecommerce', 'retail', 'manufacturing',
        'logistics', 'education', 'saas', 'enterprise', 'startup', 'agile', 'scrum'
    ],
    'tools': [
        'jira', 'confluence', 'slack', 'trello', 'asana', 'salesforce', 'sap',
        'servicenow', 'gitlab', 'github', 'bitbucket', 'swagger', 'postman'
    ]
}


def extract_mandatory_skills(job_description: str) -> list:
    """
    Extract mandatory skills from job description.
    Looks for sentences containing: "must have", "required", "mandatory", "essential"
    
    Args:
        job_description: Job description text
    
    Returns:
        List of unique mandatory skills (lowercased)
    """
    import re
    from nltk.tokenize import sent_tokenize
    
    mandatory_phrases = ['must have', 'required', 'mandatory', 'essential', 'must know']
    mandatory_keywords = set()
    
    sentences = sent_tokenize(job_description.lower())
    
    for sentence in sentences:
        # Check if sentence contains mandatory phrases
        if any(phrase in sentence for phrase in mandatory_phrases):
            # Extract keywords from this sentence
            processed = remove_stopwords(preprocess_text(sentence))
            words = word_tokenize(processed)
            mandatory_keywords.update(words)
    
    # If no explicit mandatory phrases found, extract top job keywords as fallback
    if not mandatory_keywords:
        job_keywords = extract_keywords(job_description, 20)
        mandatory_keywords = set([kw[0] for kw in job_keywords])
    
    return list(mandatory_keywords)


def calculate_category_scores(resume_text: str, job_description: str) -> dict:
    """
    Calculate skill match scores for each skill category.
    Only considers skills that are mentioned in the job description.
    
    Args:
        resume_text: Resume text
        job_description: Job description text
    
    Returns:
        Dictionary with category scores and average:
        {
            'technical': 0.75,
            'soft_skills': 0.60,
            'domain': 0.50,
            'tools': 0.40,
            'average': 0.56
        }
    """
    resume_lower = resume_text.lower()
    job_lower = job_description.lower()
    
    category_scores = {}
    
    for category, keywords in SKILL_CATEGORIES.items():
        # Count how many skills from this category are required (in job description)
        job_required_skills = [kw for kw in keywords if kw in job_lower]
        
        if job_required_skills:
            # Count how many required skills are found in resume
            resume_matches = sum(1 for skill in job_required_skills if skill in resume_lower)
            # Score = matched skills / required skills (only count what's in JD)
            score = (resume_matches / len(job_required_skills)) * 100
            category_scores[category] = round(score, 2)
        else:
            # Category not mentioned in job description - no score
            category_scores[category] = 0.0
    
    # Calculate average score (only from categories that appear in job description)
    valid_scores = [v for v in category_scores.values() if v > 0]
    average_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0
    category_scores['average'] = average_score
    
    return category_scores


def calculate_mandatory_coverage(resume_text: str, mandatory_skills: list) -> float:
    """
    Calculate percentage of mandatory skills found in resume.
    
    Args:
        resume_text: Resume text
        mandatory_skills: List of mandatory skills to check
    
    Returns:
        Coverage percentage (0-100)
    """
    resume_lower = resume_text.lower()
    found_count = sum(1 for skill in mandatory_skills if skill in resume_lower)
    
    coverage = (found_count / len(mandatory_skills)) * 100 if mandatory_skills else 0.0
    return round(coverage, 2)


def calculate_hybrid_score(
    semantic_score: float,
    category_score: float,
    mandatory_coverage: float,
    weights: dict = None
) -> float:
    """
    Calculate hybrid final score with weighted components.
    
    Args:
        semantic_score: TF-IDF Cosine similarity score (0-100)
        category_score: Average skill category score (0-100)
        mandatory_coverage: Mandatory skill coverage percentage (0-100)
        weights: Custom weights dict. Default: {'semantic': 0.5, 'category': 0.3, 'mandatory': 0.2}
    
    Returns:
        Hybrid final score (0-100)
    """
    if weights is None:
        weights = {'semantic': 0.5, 'category': 0.3, 'mandatory': 0.2}
    
    # Normalize scores to 0-100 range
    semantic_norm = semantic_score / 100.0
    category_norm = category_score / 100.0
    mandatory_norm = mandatory_coverage / 100.0
    
    hybrid_score = (
        weights['semantic'] * semantic_norm +
        weights['category'] * category_norm +
        weights['mandatory'] * mandatory_norm
    ) * 100
    
    return round(hybrid_score, 2)


def generate_decision_and_explanation(
    hybrid_score: float,
    mandatory_coverage: float,
    semantic_score: float,
    category_score: float
) -> dict:
    """
    Generate decision (Shortlist/Review/Reject) with explanation.
    
    Args:
        hybrid_score: Hybrid final score (0-100)
        mandatory_coverage: Mandatory skill coverage percentage (0-100)
        semantic_score: Semantic similarity score (0-100)
        category_score: Average category score (0-100)
    
    Returns:
        Dictionary with:
        {
            'decision': 'Shortlist' | 'Review' | 'Reject',
            'explanation': "Clear reasoning..."
        }
    """
    if hybrid_score >= 75:
        decision = 'Shortlist'
        if mandatory_coverage >= 80:
            explanation = (
                f"Excellent candidate match (Score: {hybrid_score}%). Outstanding semantic alignment "
                f"({semantic_score}%) with comprehensive mandatory skills coverage ({mandatory_coverage}%). "
                f"Strong category skills ({category_score}%) indicate well-rounded expertise. "
                f"Recommended for immediate shortlisting."
            )
        else:
            explanation = (
                f"Strong candidate match (Score: {hybrid_score}%). Excellent semantic alignment "
                f"({semantic_score}%) and solid category skills ({category_score}%), with good mandatory "
                f"coverage ({mandatory_coverage}%). Minor gaps in required skills but overall fit is clear."
            )
    
    elif 50 <= hybrid_score < 75:
        decision = 'Review'
        if mandatory_coverage >= 70:
            explanation = (
                f"Promising candidate (Score: {hybrid_score}%). Good mandatory skill coverage "
                f"({mandatory_coverage}%) and reasonable category skills ({category_score}%), but semantic "
                f"alignment ({semantic_score}%) needs improvement. Worth reviewing for specific role requirements."
            )
        else:
            explanation = (
                f"Borderline candidate (Score: {hybrid_score}%). Mixed performance across metrics: "
                f"semantic ({semantic_score}%), category skills ({category_score}%), mandatory coverage "
                f"({mandatory_coverage}%). Requires careful review to assess fit for role-specific needs."
            )
    
    else:  # < 50
        decision = 'Reject'
        if mandatory_coverage < 50:
            explanation = (
                f"Significant mismatch (Score: {hybrid_score}%). Critical gaps in mandatory skills "
                f"({mandatory_coverage}% coverage) combined with weak semantic alignment ({semantic_score}%) "
                f"and category skills ({category_score}%). Does not meet core requirements for this position."
            )
        else:
            explanation = (
                f"Insufficient alignment (Score: {hybrid_score}%). Despite some mandatory skills "
                f"({mandatory_coverage}%), overall semantic ({semantic_score}%) and category ({category_score}%) "
                f"scores are too low. Not recommended for further consideration."
            )
    
    return {
        'decision': decision,
        'explanation': explanation
    }