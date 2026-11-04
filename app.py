import re
import os
import subprocess
import streamlit as st
import spacy
import pypdf
from google import genai
from google.genai import types

# ---------------------------------------------------------
# CYBERSECURITY: PASSWORT-SCHUTZ
# ---------------------------------------------------------
st.set_page_config(page_title="NephroDoc AI", page_icon="🛡️", layout="centered")

def check_password():
    def password_entered():
        correct_pwd = st.secrets.get("APP_PASSWORD", "nephro-secure-2026!")
        if st.session_state["password"] == correct_pwd:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔐 Bitte System-Passwort eingeben:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("❌ Falsches Passwort. Erneut versuchen:", type="password", on_change=password_entered, key="password")
        st.error("Zugriff verweigert.")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ---------------------------------------------------------
# NLP SETUP (SPACY DEUTSCH)
# ---------------------------------------------------------
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("de_core_news_sm")
    except OSError:
        subprocess.run(["pip", "install", "https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl"])
        return spacy.load("de_core_news_sm")

nlp = load_nlp()

def anonymize_text(text):
    mapping = {}
    anonymized = text
    
    # A. Regex für Geburtsdaten & PLZ
    dob_pattern = r'\b\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}\b'
    dobs = re.findall(dob_pattern, anonymized)
    for i, dob in enumerate(set(dobs)):
        token = f"[GEBURTSDATUM_{i+1}]"
        mapping[token] = dob
        anonymized = anonymized.replace(dob, token)
        
    plz_pattern = r'\b\d{5}\b'
    plzs = re.findall(plz_pattern, anonymized)
    for i, plz in enumerate(set(plzs)):
        token = f"[PLZ_{i+1}]"
        mapping[token] = plz
        anonymized = anonymized.replace(plz, token)

    # B. spaCy NER (Namen & Orte)
    doc = nlp(anonymized)
    person_idx = 1
    loc_idx = 1
    
    # Wörter, die spaCy oft fälschlicherweise als Person erkennt
    ignore_words = {"geb", "geb.", "m", "w", "dr", "prof"}
    
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        cleaned_text = ent.text.strip().lower()
        if "[" in ent.text or "]" in ent.text
