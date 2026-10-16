"""Field-level extraction accuracy on synthetic cases. Run: python eval.py (needs LLM env vars)."""
import json
from app import agents, redflags

cases = json.load(open("data/cases.json", encoding="utf-8"))
hit = total = 0
for c in cases:
    d = agents.intake_agent(c["text"]).model_dump()
    for field, want in c["expect"].items():
        blob = json.dumps(d.get(field), ensure_ascii=False).lower()
        for w in (want if isinstance(want, list) else [want]):
            total += 1
            hit += w.lower() in blob
    if c.get("expect_flags"):
        total += 1
        hit += bool(redflags.check(c["text"]))
print(f"Field accuracy: {hit}/{total} = {100*hit/total:.0f}% on {len(cases)} synthetic cases")
