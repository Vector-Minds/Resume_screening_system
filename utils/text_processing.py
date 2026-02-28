import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text: str) -> str:
    sw = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered = [w for w in words if w.lower() not in sw]
    return " ".join(filtered)