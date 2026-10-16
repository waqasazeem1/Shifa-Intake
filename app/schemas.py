from pydantic import BaseModel, Field, field_validator


class Intake(BaseModel):
    chief_complaint: str = ""
    duration: str = ""
    symptoms: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)  # fields patient did not state clearly
    patient_language: str = ""

    @field_validator("chief_complaint", "duration", "patient_language", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join(str(x) for x in v if x is not None)
        return str(v).strip()

    @field_validator("symptoms", "medications", "allergies", "history", "uncertain_fields", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if x is not None and str(x).strip()]
        return [str(v).strip()]
