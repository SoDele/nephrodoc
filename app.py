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
st.set_page_config(page_title="NephroDoc AI (Pro)", page_icon="🛡️", layout="centered")

def check_password():
    def password_entered():
        correct_pwd = st.secrets.get("APP_PASSWORD", "nephro-secure-2026!")
        # Sicherer Zugriff mit .get(), falls der Key nicht existiert
        if st.session_state.get("password", "") == correct_pwd:
            st.session_state["password_correct"] = True
            # Optional: Statt löschen einfach leeren oder sicher prüfen
            if "password" in st.session_state:
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
    
    # 1. ERWEITERTE REGEX-MUSTER FÜR MEDIZINISCHE STRUKTUREN
    
    # A. Geburtsdaten (z.B. 05.10.1989 oder 5.10.89)
    dob_pattern = r'\b\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}\b'
    for i, dob in enumerate(set(re.findall(dob_pattern, anonymized))):
        token = f"[GEBURTSDATUM_{i+1}]"
        mapping[token] = dob
        anonymized = anonymized.replace(dob, token)
        
    # B. Postleitzahlen (5-stellige Zahlen)
    plz_pattern = r'\b\d{5}\b'
    for i, plz in enumerate(set(re.findall(plz_pattern, anonymized))):
        token = f"[PLZ_{i+1}]"
        mapping[token] = plz
        anonymized = anonymized.replace(plz, token)

    # C. Kliniken, Praxen & Institutionen (z.B. "Klinikum ...", "Praxis Dr. ...", "MVZ ...")
    institution_patterns = [
        r'\b(?:Klinikum|Krankenhaus|Praxis|Gemeinschaftspraxis|MVZ|Dialysezentrum|Zentrum für)\s+[A-Zäöüß][a-zäöüß]+(?:\s+[A-Zäöüß][a-zäöüß]+)*\b',
        r'\bDr\.\s+(?:med\.\s+)?[A-Zäöüß][a-zäöüß]+(?:\s+[A-Zäöüß][a-zäöüß]+)?\b',
        r'\bProf\.\s+(?:Dr\.\s+)?[A-Zäöüß][a-zäöüß]+(?:\s+[A-Zäöüß][a-zäöüß]+)?\b'
    ]
    
    inst_idx = 1
    for pattern in institution_patterns:
        matches = re.findall(pattern, anonymized)
        for match in set(matches):
            # Nicht matchen, wenn es schon ein Token ist
            if "[" in match:
                continue
            token = f"[INSTITUTION_{inst_idx}]"
            mapping[token] = match
            anonymized = anonymized.replace(match, token)
            inst_idx += 1

    # 2. SPACY NER FÜR RESTLICHE PERSONEN & ORTE
    doc = nlp(anonymized)
    person_idx = 1
    loc_idx = 1
    
    ignore_words = {"geb", "geb.", "m", "w", "dr", "prof", "patient", "pat", "herr", "frau"}
    
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        cleaned_text = ent.text.strip().lower()
        if "[" in ent.text or "]" in ent.text or "_" in ent.text or cleaned_text in ignore_words:
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

# ---------------------------------------------------------
# GEMINI GENERIERUNG
# ---------------------------------------------------------
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
    - Der Text enthält anonymisierte Tokens in eckigen Klammern (z. B. [PERSON_1], [GEBURTSDATUM_1], [INSTITUTION_1]).
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
        'gemini-3.5-flash-lite'
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
# BENUTZEROBERFLÄCHE (STREAMLIT)
# ---------------------------------------------------------
st.title("🛡️ NephroDoc AI (Pro Edition)")
st.caption("Erweiterte Hybrid-Anonymisierung (Regex + NER) vor Cloud-Übergabe")

api_key_input = st.text_input("Google AI Studio API Key:", type="password", value=os.getenv("GEMINI_API_KEY", ""))

st.divider()

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
                anonymized_text, mapping = anonymize_text(transcript)
                
                st.subheader("🔍 Erweitertes Anonymisierungs-Protokoll (Mapping)")
                if mapping:
                    for token, original in mapping.items():
                        st.text(f"{token} ➔ {original}")
                else:
                    st.info("Keine sensiblen Daten erkannt.")
                
                raw_letter, active_model = generate_nephro_letter_gemini(api_key_input, anonymized_text, lab_data_manual)
                final_letter = deanonymize_text(raw_letter, mapping)
                
                st.subheader("3. Fertiger Nephrologischer Arztbrief (Re-Anonymisiert)")
                st.code(final_letter, language="markdown")
                st.success(f"Erfolgreich generiert mit Modell: {active_model}")
                
            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung: {e}")
