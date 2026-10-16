# Text for the lablab submission form

**Challenge:** Health and Wellbeing

**Title:** Shifa Intake: Safe Urdu-first patient intake with human review

**Short description:** Roman Urdu/Urdu/English patient intake and instruction translation for clinics, running on AMD GPUs. Never diagnoses, flags urgent cases by rules, and needs staff approval at every step.

**Long description:**
Clinic staff in Pakistan spend time turning patients' Urdu or Roman Urdu descriptions into structured English notes, and patients often cannot understand English discharge instructions.
Shifa Intake takes a patient's free-text message, extracts structured fields (complaint, duration, symptoms, medicines, allergies, history) with an open LLM served on AMD infrastructure, and writes a short doctor summary that lists what is unclear instead of guessing.
Urgent signs (chest pain, breathing difficulty, self-harm language, etc.) are detected by transparent rules, not the model. An output guardrail blocks any text that looks like diagnosis or treatment advice.
Staff review and correct the fields, the doctor writes instructions in English, the system drafts a simple Roman Urdu version, and staff approve it before release. The API enforces this order.
We tested field extraction on 6 synthetic cases (accuracy printed by eval.py). All data is synthetic.

**Tags:** AMD, ROCm, LLM, healthcare, multilingual, FastAPI, human-in-the-loop

**Demo URL:** https://substantially-marks-hat-iowa.trycloudflare.com

**Who uses it:** clinic reception/nursing staff and doctors.
