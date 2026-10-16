"""Rule-based red-flag check. Deliberately NOT an LLM: urgent cases must never depend on model output."""
import re

RULES = [
    (r"chest pain|seene (mein|me) dard|سینے", "Chest pain reported"),
    (r"short(ness)? of breath|can'?t breathe|saans (nahi|ki takleef|phool)|سانس", "Breathing difficulty reported"),
    (r"unconscious|faint|behosh|بے ہوش", "Loss of consciousness reported"),
    (r"heavy bleeding|vomiting blood|khoon (ki ulti|bahut)|blood in (stool|urine)", "Significant bleeding reported"),
    (r"stroke|face droop|slurred|zuban (larkha|ruk)|ek taraf kamzori|weakness on one side", "Possible stroke signs reported"),
    (r"suicid|kill myself|khudkushi|jaan dena|مرنا چاہتا", "Self-harm risk language reported"),
    (r"seizure|daura|convulsion|mirgi", "Seizure reported"),
    (r"pregnan.*(bleed|pain)|hamila.*(khoon|dard)", "Pregnancy with bleeding/pain reported"),
]


def check(text: str) -> list[str]:
    t = text.lower()
    return [label for pat, label in RULES if re.search(pat, t)]
