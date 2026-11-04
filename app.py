import re
import os
import streamlit as st
import spacy
import requests
import json
import pypdf

# ---------------------------------------------------------
# NLP SETUP (SPACY DEUTSCH)
# ---------------------------------------------------------
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("de_core_news_sm")
    except OSError:
        import subprocess
        subprocess.run(["pip", "install", "https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl"])
        return spacy.load("de_core_news_sm")

nlp = load_nlp()

# ---------------------------------------------------------
# PAGE CONFIG & AUTHENTICATION
# ---------------------------------------------------------
st.set_page_config(page_title="NephroDoc AI", page_icon="🛡️", layout="centered")

# Einfacher Passwortschutz für das Web (Konfigurierbar über Streamlit Secrets oder direkt hier)
def check_password():
    """Returns True if the user entered the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", "nephro2026"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Bitte Passwort eingeben, um die App zu entsperren:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Falsches Passwort. Bitte erneut versuchen:", type="password", on_change=password_entered, key="password")
        st.error("😕 Falsches Passwort")
        return False
    else:
        return True

if not check_password():
    st.stop()

# ---------------------------------------------------------
# SPACY NLP SETUP (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_nlp():
    return spacy.load("de_core_news_sm")

nlp = load_nlp()

def anonymize_text(text):
    mapping = {}
    anonymized = text
    
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

    doc = nlp(anonymized)
    person_idx = 1
    loc_idx = 1
    
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if "[" in ent.text or "]" in ent.text or "_" in ent.text:
            continue
        if ent.label_ == "PER":
            token = f"[PERSON_{person_idx}]"
            mapping[token] = ent.text
            anonymized = anonymized[:ent.start_char] + token + anonymized[ent.end_char:]
            person_idx += 1
        elif ent.label_ == "LOC":
            token = f"[ORT_{loc_idx}]"
            mapping[token] = ent.text
            anonymized = anonymized[:ent.start_char] + token + anonymized[ent.end_char:]
            loc_idx += 1
            
    return anonymized, mapping

def deanonymize_text(text, mapping):
    result = text
    for token, original_value in mapping.items():
        result = result.replace(token, original_value)
        raw_token = token.strip("[]")
        result = re.sub(rf'\b{re.escape(raw_token)}\b', original_value, result)
    return result

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = pypdf.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Fehler beim PDF-Lesen: {e}")
    return text

def generate_nephro_letter_gemini(api_key, anonymized_transcript, lab_data):
    client = genai.Client(api_key=api_key)
    
    system_instruction = """
    Du bist ein hochqualifizierter medizinischer Fachassistent für Nephrologie und Dialyse. 
    Erstelle aus den bereitgestellten Visiten-Notizen und Labor-/Dialysedaten einen professionellen, 
    strukturierten nephrologischen Arztbrief auf Deutsch.
    
    Gliederung des Briefes:
    1. Anamnese & Aktueller Status
    2. Dialysestatus / Therapie (falls vorhanden)
    3. Labor- & Befundbewertung
    4. Zusammenfassende Beurteilung & Empfehlungen (Prozedere)
    
    STRIKTE DATENSCHUTZ- UND PLATZHALTER-REGELN:
    - Der Text enthält anonymisierte Tokens in eckigen Klammern (z. B. [PERSON_1], [GEBURTSDATUM_1], [ORT_1]).
    - Du MUSST diese exakten Platzhalter im generierten Brief WORTWÖRTLICH an den passenden Stellen beibehalten.
    - Erfinde KEINE Fantasienamen und füge keine echten Personennamen ein. 
    - Gib ausschließlich den fertig formatierten Arztbrief zurück.
    """
    
    user_content = f"""
    [LABOR- UND DIALYSEDATEN]
    {lab_data if lab_data.strip() else "Keine gesonderten Labordaten angegeben."}
    
    [VISITEN-TRANSKRIPT]
    {anonymized_transcript}
    """

    candidate_models = [
        'gemini-3.8-flash',
        'gemini-3.6-flash',
        'gemini-3.5-flash-lite',
        'gemini-3.1-pro-preview'
    ]

    last_exception = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
            )
            return response.text, model_name
        except Exception as e:
            last_exception = e

    raise RuntimeError(f"Keines der verfügbaren Gemini-Modelle konnte erreicht werden. Letzter Fehler: {last_exception}")

# ---------------------------------------------------------
# WEB UI (STREAMLIT)
# ---------------------------------------------------------
st.title("🛡️ NephroDoc AI (Web Edition)")
st.caption("Privacy-by-Design: Lokale Anonymisierung vor Cloud-Übergabe an Gemini")

# API Key Eingabe (Seitenleiste oder Hauptfenster)
api_key_input = st.text_input("Google AI Studio API Key:", type="password", value=os.getenv("GEMINI_API_KEY", ""))

st.divider()

# Eingabefelder
transcript = st.text_area("1. Visiten-Transkript (aus MacWhisper einfügen):", height=150)

uploaded_file = st.file_uploader("2. Laborwerte / Dialyseprotokoll (PDF hochladen):", type=["pdf"])
lab_data_from_pdf = ""
if uploaded_file is not None:
    lab_data_from_pdf = extract_text_from_pdf(uploaded_file)
    st.success(f"PDF '{uploaded_file.name}' erfolgreich ausgelesen!")

lab_data_manual = st.text_area("Oder Labordaten manuell eingeben/ergänzen:", value=lab_data_from_pdf, height=100)

if st.button("🚀 Anonymisieren & Arztbrief generieren", type="primary"):
    if not api_key_input:
        st.error("Bitte gib deinen Google AI Studio API-Key ein.")
    elif not transcript:
        st.error("Das Visiten-Transkript darf nicht leer sein.")
    else:
        with st.spinner("Verarbeite Daten und generiere Arztbrief..."):
            try:
                # 1. Anonymisieren
                anonymized_text, mapping = anonymize_text(transcript)
                
                # 2. Mapping anzeigen
                st.subheader("🔍 Anonymisierungs-Protokoll (Mapping)")
                if mapping:
                    for token, original in mapping.items():
                        st.text(f"{token} ➔ {original}")
                else:
                    st.info("Keine sensiblen Daten erkannt.")
                
                # 3. Gemini Aufruf
                raw_letter, active_model = generate_nephro_letter_gemini(api_key_input, anonymized_text, lab_data_manual)
                
                # 4. Re-Anonymisieren
                final_letter = deanonymize_text(raw_letter, mapping)
                
                # 5. Output
                st.subheader("3. Fertiger Nephrologischer Arztbrief (Re-Anonymisiert)")
                st.code(final_letter, language="markdown")
                st.success(f"Erfolgreich generiert mit Modell: {active_model}")
                
            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung: {e}")
