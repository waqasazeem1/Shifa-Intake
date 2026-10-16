# Shifa Intake (AMD Developer Hackathon: ACT III, Health and Wellbeing)

Patient intake and multilingual (Urdu / Roman Urdu / English) communication assistant for clinic staff.
It never diagnoses or advises treatment. Every output is reviewed by a human before it reaches the patient.

## Flow
1. Patient message (Urdu / Roman Urdu / English) -> **Intake agent** extracts structured fields (LLM on AMD).
2. **Rule-based red-flag check** (not LLM) shows urgent warnings (chest pain, breathing trouble, self-harm, etc.).
3. **Summary agent** writes a doctor summary and lists unclear fields as "uncertain". An **output guardrail** blocks diagnosis/treatment-like text.
4. **Staff review**: staff corrects fields and approves.
5. Doctor writes instructions in English -> **Translation agent** gives Roman Urdu draft -> staff approves -> released.

## Safety
- No diagnosis, no medicine or dose suggestions (prompt rules + regex output guardrail)
- Uncertainty is shown per field; model failure routes the case to manual review (HTTP 502 message)
- Human approval is enforced by the API (cannot translate before intake approval, cannot release before translation approval)
- Synthetic data only. Do not enter real patient data.

## AMD
The LLM runs on AMD infrastructure through an OpenAI-compatible endpoint (AMD tokenfactory or vLLM on AMD Developer Cloud with ROCm). Set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # fill values, then export them (or set in your shell)
uvicorn app.main:app --reload
python eval.py         # field accuracy on 6 synthetic cases
```
Docker: `docker build -t shifa . && docker run -p 8000:8000 --env-file .env shifa`
