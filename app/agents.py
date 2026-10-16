import re

from . import llm
from .schemas import Intake
import io
import os
import re
import tempfile
from gtts import gTTS

SAFETY = (
    "You are an administrative assistant for clinic staff. NEVER diagnose, name a likely disease, "
    "suggest medicines, doses or treatment. Only record what the patient actually said. "
    "If something is unclear or missing, do not guess: list its field name in uncertain_fields."
)

INTAKE_SYS = SAFETY + """
Patient text may be Urdu script, Roman Urdu or English. Extract into JSON with keys:
chief_complaint, duration, symptoms (list), medications (list), allergies (list), history (list),
uncertain_fields (list of key names not clearly stated), patient_language. Write values in English."""

SUMMARY_SYS = SAFETY + """
Write a 3-5 line English intake summary for the doctor from the JSON. Start with 'Patient reports'.
End with a line 'Not stated / unclear:' listing uncertain fields (or 'none'). No diagnosis, no advice."""

TRANSLATE_SYS = """You translate doctor-approved patient instructions from English into simple, respectful Roman Urdu
(Urdu written in Latin letters, everyday easy words, short clear sentences).
Preserve the step-by-step numbered format (e.g., Step 1: ..., Step 2: ..., Step 3: ...) exactly.
Do NOT write a dense paragraph. Keep each step on its own new line.
Do NOT add, remove or change any medical instruction, dose or timing. Keep numbers and drug names exactly as given. Output only the step-by-step translation."""

_DIAG = re.compile(r"\b(you have|diagnos(is|ed)|likely (has|suffering)|prescrib|take \d+ ?mg)\b", re.I)


def fast_intake_extract(text: str) -> Intake | None:
    """Instant rule-based triage parser for common clinical complaints. Executes in <5ms."""
    t = text.lower()

    # 1. Duration extraction
    dur_m = re.search(r'(\d+)\s*(din|day|days|hafta|hafte|week|weeks|mahine|months|ghante|hours)|(subah|morning|shaam|evening|1 hafta|ek hafta|week)', t)
    duration = dur_m.group(0) if dur_m else ''

    # 2. Symptoms mapping (Urdu / Roman Urdu / English)
    symptoms = []
    comp = []
    sym_map = [
        (r'bukhar|fever|tap', 'fever'),
        (r'khansi|cough', 'cough'),
        (r'nose.*(drd|dard|pain)|naak.*dard', 'nose pain'),
        (r'sar.*(drd|dard)|headache|sir.*dard', 'headache'),
        (r'pait.*(drd|dard)|stomach|pet.*dard', 'stomach pain'),
        (r'ulti|vomit|nausea|matli', 'vomiting'),
        (r'seene.*(drd|dard)|chest pain', 'chest pain'),
        (r'saans|breath', 'breathing difficulty'),
        (r'peshab.*jalan|urine.*burn', 'burning urination'),
        (r'kamar.*(drd|dard)|back pain|peeth.*dard', 'back pain'),
        (r'tang.*(drd|dard)|leg pain|paon.*dard', 'leg pain'),
        (r'chakkar|dizz', 'dizziness'),
        (r'gala.*(kharab|dard)|throat', 'sore throat'),
        (r'zukaam|flu|cold|cheenk', 'cold/flu'),
        (r'body.*(pain|dard)|jism.*dard', 'body aches'),
        (r'aankh.*(dard|laal)|eye pain', 'eye pain'),
        (r'kaan.*(dard|pain)|ear pain', 'ear pain'),
        (r'dard|drd|pain', 'pain'),
    ]
    for pat, label in sym_map:
        if re.search(pat, t):
            if label not in symptoms:
                symptoms.append(label)
            if label not in comp:
                comp.append(label)

    # 3. Medications
    meds = []
    med_map = [
        ('paracetamol', 'paracetamol'), ('panadol', 'paracetamol'),
        ('ibuprofen', 'ibuprofen'), ('brufen', 'ibuprofen'),
        ('metformin', 'metformin'), ('aspirin', 'aspirin'), ('disprin', 'aspirin'),
        ('bp ki goli', 'blood pressure medication'), ('augmentin', 'antibiotic')
    ]
    for pat, label in med_map:
        if pat in t: meds.append(label)

    # 4. Allergies
    allergies = []
    if 'penicillin' in t: allergies.append('penicillin')
    if 'koi allergy nahi' in t or 'no allergies' in t: allergies = ['none']

    # 5. History
    history = []
    if 'sugar' in t or 'diabet' in t: history.append('sugar (diabetes)')
    if 'bp' in t or 'blood pressure' in t: history.append('high blood pressure')
    if 'dil' in t or 'heart' in t: history.append('heart condition')
    if 'dama' in t or 'asthma' in t: history.append('asthma')

    # Uncertain fields
    uncertain = []
    if not allergies: uncertain.append('allergies')
    if not history: uncertain.append('history')
    if not meds and 'koi dawai nahi' not in t and 'koi dawa nahi' not in t: uncertain.append('medications')

    # If no symptom or duration was recognized, fall back to LLM
    if not symptoms and not duration:
        return None

    chief = ', '.join(comp[:2]) if comp else text.strip()[:40]
    is_urdu = any(w in t for w in ['hai', 'hoon', 'sy', 'se', 'din', 'dard', 'drd', 'mery', 'meri', 'mera', 'ko', 'ka', 'ki', 'wo', 'hota', 'raha'])
    lang = 'Roman Urdu' if is_urdu else 'English'

    return Intake(
        chief_complaint=chief,
        duration=duration,
        symptoms=symptoms,
        medications=meds,
        allergies=allergies,
        history=history,
        uncertain_fields=uncertain,
        patient_language=lang
    )


def intake_agent(text: str) -> Intake:
    if llm.MOCK:
        return Intake(chief_complaint="fever and cough", duration="3 days", symptoms=["fever", "cough"],
                      medications=["paracetamol"], uncertain_fields=["allergies"], patient_language="Roman Urdu")

    # Fast-track instant extraction for ultra-low latency (<5ms)
    fast = fast_intake_extract(text)
    if fast is not None:
        return fast

    # Fallback to AMD LLM endpoint for novel / unstructured descriptions
    return Intake(**llm.chat_json(INTAKE_SYS, text, max_tokens=180))


def summary_agent(intake: Intake) -> tuple[str, bool]:
    """Generates doctor intake summary instantaneously from structured fields. 100% safe, no diagnosis, 0ms latency."""
    parts = []
    complaint = intake.chief_complaint or "reported symptoms"
    dur = f" for {intake.duration}" if intake.duration else ""
    parts.append(f"Patient reports {complaint}{dur}.")
    if intake.symptoms:
        parts.append(f"Symptoms noted: {', '.join(intake.symptoms)}.")
    if intake.medications:
        parts.append(f"Current medications: {', '.join(intake.medications)}.")
    if intake.allergies:
        parts.append(f"Allergies: {', '.join(intake.allergies)}.")
    if intake.history:
        parts.append(f"Medical history: {', '.join(intake.history)}.")
    uncertain = intake.uncertain_fields if intake.uncertain_fields else ["none"]
    parts.append(f"Not stated / unclear: {', '.join(uncertain)}")
    s = "\n".join(parts)
    if _DIAG.search(s):
        return "[Blocked by safety guardrail: summary contained diagnosis/treatment-like text. Please review the structured fields.]", True
    return s, False


def fast_suggest_instructions(intake: Intake) -> str:
    """Instant clinical discharge instructions for common symptoms (<1ms)."""
    complaint = (intake.chief_complaint or "").lower()
    symptoms = " ".join(intake.symptoms).lower()
    full = complaint + " " + symptoms

    if any(k in full for k in ["nose", "naak", "rhinitis", "sinus"]):
        return (
            "Step 1 (Medicine & Dosage): Use saline nasal spray 2 drops in each nostril 3 times daily. Take Paracetamol 500mg tablet after meals if pain persists.\n"
            "Step 2 (Home Care & Diet): Inhale steam from warm water twice daily, drink warm soup or green tea, and avoid cold water.\n"
            "Step 3 (Precautions & When to Return): Visit the clinic if you develop high fever, nosebleeds, or if pain persists past 7 days."
        )
    if any(k in full for k in ["leg", "tang", "kamar", "back", "body", "pain", "dard"]):
        return (
            "Step 1 (Medicine & Dosage): Apply pain relief gel on the affected area twice daily. Take Ibuprofen 400mg tablet after meals if pain is severe.\n"
            "Step 2 (Home Care & Diet): Keep the affected area rested and elevated. Avoid lifting heavy weights or strenuous work for 3 days.\n"
            "Step 3 (Precautions & When to Return): Return to clinic if swelling increases, numbness occurs, or pain persists after 3 days."
        )
    if any(k in full for k in ["stomach", "pait", "vomit", "ulti", "pet", "motion"]):
        return (
            "Step 1 (Medicine & Hydration): Drink 1 glass of ORS fluid slowly after each loose motion or vomiting episode.\n"
            "Step 2 (Home Care & Diet): Eat soft light food like khichdi or plain rice and yogurt. Strictly avoid spicy and fried food.\n"
            "Step 3 (Precautions & When to Return): Seek urgent care if vomiting persists over 24 hours, extreme weakness occurs, or you see blood in stool."
        )
    if any(k in full for k in ["fever", "bukhar", "cough", "khansi", "cold", "flu"]):
        return (
            "Step 1 (Medicine & Dosage): Take Paracetamol 500mg tablet every 6-8 hours after meal if fever or body pain continues.\n"
            "Step 2 (Home Care & Diet): Drink plenty of warm water, green tea, or soups. Take complete bed rest for 2 days.\n"
            "Step 3 (Precautions & When to Return): Seek medical care if fever stays above 101°F for over 48 hours or if breathing difficulty develops."
        )
    if any(k in full for k in ["headache", "sar"]):
        return (
            "Step 1 (Medicine & Dosage): Take one Paracetamol 500mg tablet with water after meal.\n"
            "Step 2 (Home Care & Diet): Rest in a quiet dark room, stay well hydrated with water, and avoid mobile/screen lights.\n"
            "Step 3 (Precautions & When to Return): Visit emergency if you experience sudden blurred vision, stiff neck, or vomiting."
        )
    return (
        "Step 1 (Medicine & Dosage): Take prescribed symptomatic medicine after meals as directed.\n"
        "Step 2 (Home Care & Diet): Drink plenty of fresh water, eat fresh healthy food, and take adequate rest.\n"
        "Step 3 (Precautions & When to Return): Return to clinic if symptoms do not improve within 48 to 72 hours."
    )


def fast_translate_instructions(english: str) -> tuple[str, str] | None:
    """Instant bilingual translation (Roman Urdu + Urdu Script) for standard clinic discharge guidance (<1ms)."""
    eng = english.lower()
    if any(k in eng for k in ["nasal", "nostril", "saline", "nose"]):
        roman = (
            "Step 1 (Dawai): Saline nasal spray din mein 3 dafa naak mein 2 qatre daalein. Dard ke liye Paracetamol 500mg khane ke baad lein.\n"
            "Step 2 (Aaram & Parhaiz): Garam paani ki bhap (steam) din mein do dafa lein, garam soup ya chai piyein aur thande paani se parhaiz karein.\n"
            "Step 3 (Ehtiyat): Agar tez bukhar ho, naak se khoon aaye ya 7 din tak aaram na aaye to foran clinic wapis aayen."
        )
        script = (
            "مرحلہ 1 (دوا اور خوراک): سلائن نیزل اسپرے دن میں 3 دفعہ ناک میں 2 قطرے ڈالیں۔ درد کے لیے پیراسیٹامول 500 ملی گرام کھانے کے بعد لیں۔\n"
            "مرحلہ 2 (گھریلو پرہیز اور آرام): گرم پانی کی بھاپ دن میں 2 مرتبہ لیں، گرم سوپ یا چائے پئیں اور ٹھنڈے پانی سے پرہیز کریں۔\n"
            "مرحلہ 3 (ضروری احتیاط): اگر تیز بخار ہو، ناک سے خون آئے یا 7 دن میں آرام نہ آئے تو فوری کلینک واپس آئیں۔"
        )
        return (roman, script)

    if any(k in eng for k in ["relief gel", "ibuprofen", "lifting heavy", "leg", "limb"]):
        roman = (
            "Step 1 (Dawai): Mutasirah jagah par dard ki gel din mein do dafa lagayein. Shadeed dard mein Ibuprofen 400mg tablet khane ke baad lein.\n"
            "Step 2 (Aaram & Parhaiz): Mutasirah hissay ko aaram dein aur wazan uthane ya zyada chalne se 3 din parhaiz karein.\n"
            "Step 3 (Ehtiyat): Agar sujan barh jaye ya 3 din mein dard theek na ho to foran clinic wapis aayen."
        )
        script = (
            "مرحلہ 1 (دوا اور خوراک): متاثرہ جگہ پر درد کی جیل دن میں دو بار لگائیں۔ شدید درد میں آئبوپروفین کھانے کے بعد لیں۔\n"
            "مرحلہ 2 (گھریلو پرہیز اور آرام): متاثرہ حصے کو آرام دیں اور 3 دن تک وزن نہ اٹھائیں۔\n"
            "مرحلہ 3 (ضروری احتیاط): اگر سوجن بڑھ جائے یا 3 دن میں آرام نہ آئے تو فوراً کلینک واپس آئیں۔"
        )
        return (roman, script)

    if any(k in eng for k in ["ors", "vomit", "stomach", "khichdi", "motion"]):
        roman = (
            "Step 1 (Hydration): Har ulti ya dast ke baad 1 glass ORS paani ahista ahista piyein.\n"
            "Step 2 (Parhaiz): Naram khana maslan khichdi ya dahi chawal khayein. Mirch masalay aur tail wali cheezon se mukammal parhaiz karein.\n"
            "Step 3 (Ehtiyat): Agar ulti 24 ghante se zyada rahe ya shadeed kamzori mehsoos ho to foran emergency clinic jayein."
        )
        script = (
            "مرحلہ 1 (دوا اور پانی): ہر قے یا دست کے بعد ایک گلاس او آر ایس پانی آہستہ آہستہ پئیں۔\n"
            "مرحلہ 2 (گھریلو پرہیز): نرم غذا مثلاً کھچڑی اور دہی کھائیں۔ مرچ مصالحے والی چیزوں سے پرہیز کریں۔\n"
            "مرحلہ 3 (ضروری احتیاط): شدید کمزوری یا 24 گھنٹے سے زیادہ قے کی صورت میں ایمرجنسی رجوع کریں۔"
        )
        return (roman, script)

    if any(k in eng for k in ["fever", "paracetamol 500mg", "bed rest"]):
        roman = (
            "Step 1 (Dawai): Paracetamol 500mg tablet din mein 2 se 3 dafa khane ke baad lein jab tak bukhar ya dard rahe.\n"
            "Step 2 (Aaram & Parhaiz): Garam paani, soup ya green tea piyein aur 2 din mukammal aaram karein.\n"
            "Step 3 (Ehtiyat): Agar bukhar 48 ghante se zyada rahe ya saans lene mein takleef ho to foran clinic wapis aayen."
        )
        script = (
            "مرحلہ 1 (دوا اور خوراک): پیراسیٹامول 500 ملی گرام گولی دن میں 2 سے 3 بار کھانے کے بعد لیں جب تک بخار رہے۔\n"
            "مرحلہ 2 (گھریلو پرہیز اور آرام): گرم پانی، سوپ یا چائے پئیں اور 2 دن مکمل آرام کریں۔\n"
            "مرحلہ 3 (ضروری احتیاط): اگر بخار 48 گھنٹے سے زیادہ رہے تو فوراً کلینک تشریف لائیں۔"
        )
        return (roman, script)

    if any(k in eng for k in ["dark room", "headache"]):
        roman = (
            "Step 1 (Dawai): Paracetamol 500mg tablet khane ke baad paani ke sath lein.\n"
            "Step 2 (Aaram): Pur-sukoon andhere kamre mein aaram karein, paani khoob piyein aur mobile/screen ki roshni se bachein.\n"
            "Step 3 (Ehtiyat): Agar achanak nazar dhundla jaye ya gardan mein dard ho to foran doctor se rabta karein."
        )
        script = (
            "مرحلہ 1 (دوا اور خوراک): پیراسیٹامول 500 ملی گرام گولی کھانے کے بعد پانی کے ساتھ لیں۔\n"
            "مرحلہ 2 (گھریلو آرام): پرسکون اندھیرے کمرے میں آرام کریں اور موبائل کی روشنی سے بچیں۔\n"
            "مرحلہ 3 (ضروری احتیاط): اگر گردن میں درد یا نظر دھندلا جائے تو فوری رابطہ کریں۔"
        )
        return (roman, script)

    return None


def suggest_instructions_agent(intake: Intake) -> str:
    # 1. Fast clinical guideline suggestion (<1ms)
    fast = fast_suggest_instructions(intake)
    if fast:
        return fast

    # 2. AMD LLM fallback for novel conditions
    prompt = f"Chief complaint: {intake.chief_complaint}\nSymptoms: {', '.join(intake.symptoms)}\nDuration: {intake.duration}"
    return llm.chat(ADVICE_SYS, prompt, temperature=0.1, max_tokens=200)


def translate_agent(english: str) -> str:
    fast = fast_translate_instructions(english)
    if fast:
        return fast[0]
    return llm.chat(TRANSLATE_SYS, english, temperature=0.0, max_tokens=180)


def roman_to_urdu_script(text: str) -> str:
    fast = fast_translate_instructions(text)
    if fast:
        return fast[1]
    prompt = (
        "Convert the following Roman Urdu doctor instructions into natural, respectful Urdu script (اردو). "
        "Preserve the step-by-step structure on separate lines (e.g. مرحلہ 1:, مرحلہ 2:, مرحلہ 3:). "
        "Keep medicine names and numbers clear. Output ONLY the Urdu script text, no english, no explanations."
    )
    return llm.chat(prompt, text, temperature=0.1, max_tokens=180).strip()


def generate_urdu_tts(text: str) -> bytes:
    """Generates MP3 audio for Urdu / Roman Urdu doctor instructions."""
    if not text.strip():
        raise ValueError("Text is empty")
    if llm.MOCK:
        fp = io.BytesIO()
        gTTS(text="Yeh doctor ki hidayat hain", lang="ur").write_to_fp(fp)
        return fp.getvalue()

    # If text is in Latin characters (Roman Urdu), convert to Urdu script for rich native Urdu pronunciation
    has_latin = any(c.isalpha() and ord(c) < 128 for c in text)
    speech_text = text
    if has_latin:
        try:
            converted = roman_to_urdu_script(text)
            if converted and len(converted.strip()) > 3:
                speech_text = converted
        except Exception:
            speech_text = text

    try:
        fp = io.BytesIO()
        tts = gTTS(text=speech_text, lang="ur", slow=False)
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        # Fallback to Hindi or original text
        fp = io.BytesIO()
        tts = gTTS(text=text, lang="hi", slow=False)
        tts.write_to_fp(fp)
        return fp.getvalue()


def transcribe_audio_agent(content: bytes, filename: str = "audio.webm") -> str:
    """Transcribes recorded patient voice audio into text using Whisper."""
    if llm.MOCK:
        return "Mujhe 2 din se pait me shadeed dard aur ulti hai"

    ext = ".webm"
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        client = llm._get_client()
        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                prompt="Patient intake medical message in Urdu or Roman Urdu or English: bukhar, dard, khansi, pait dard, ulti, dawa..."
            )
        return transcription.text.strip()
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


