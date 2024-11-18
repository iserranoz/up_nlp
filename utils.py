import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer

# Cargar modelo spaCy para inglés
#nlp = spacy.load("en_core_web_sm")

def setup_nltk():
    """Descarga los recursos necesarios de NLTK."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

def normalize_text(doc):
    """Normaliza el texto eliminando HTML, URLs, caracteres especiales y números."""
    doc = doc.lower()
    doc = BeautifulSoup(doc, "html.parser").get_text()
    doc = re.sub(r'https?://\S+|www\.\S+', ' ', doc)  # Eliminar URLs
    doc = re.sub(r'[^\w\s]', ' ', doc)  # Eliminar caracteres especiales
    doc = re.sub(r'\d+', ' ', doc)  # Eliminar números
    doc = re.sub(r'\s+', ' ', doc).strip()  # Eliminar espacios extra
    return doc.strip()

def transform_text(texts):
    """Normaliza una lista de textos."""
    return [normalize_text(text) for text in texts]

def remove_stopwords(tokens):
    """Elimina las stopwords de una lista de tokens."""
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def tokenize_text(normalized_text):
    """Tokeniza el texto normalizado, elimina stopwords y aplica lematización."""
    # Tokenización
    tokens = nltk.word_tokenize(normalized_text)
    # Eliminar stopwords
    tokens = remove_stopwords(tokens)
    return tokens

def create_bow_features(train_corpus, test_corpus, ngram_range=(1, 3), min_df=0.0, max_df=1.0):
    """Crea características Bag of Words (BOW) para los corpus de entrenamiento y prueba."""
    vectorizer = CountVectorizer(binary=False, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    train_features = vectorizer.fit_transform(train_corpus)
    test_features = vectorizer.transform(test_corpus)
    return vectorizer, train_features, test_features

setup_nltk()
