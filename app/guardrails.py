"""Input/output guardrails: prompt injection, jailbreaks, blocked topics, PII."""
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules|prompts)",
    r"disregard (all |any )?(previous|prior|your) (instructions|rules)",
    r"(reveal|show|print|repeat) (your |the )?(system|hidden|initial) (prompt|instructions)",
    r"you are now (dan|an? unrestricted|in developer mode)",
    r"\bjailbreak\b|\bdan mode\b|developer mode enabled",
    r"pretend (you have|there are) no (rules|restrictions|guidelines)",
    r"act as (an? )?(unfiltered|unrestricted) ",
]
BLOCKED_TOPICS = {
    "weapons": r"\b(make|build|create)\b.*\b(bomb|explosive|weapon)\b",
    "medical_advice": r"\b(diagnos\w+|prescribe|dosage)\b",
    "legal_advice": r"\b(sue|lawsuit|legal advice)\b",
    "politics": r"\b(election|prime minister|political party)\b",
}
PII_PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "PHONE": r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)",
    "CARD": r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)",
    "AADHAAR": r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)",
}
# addresses we are allowed to print (public support contact)
ALLOWED_EMAILS = {"support@homenest.example"}


def check_input(text: str) -> tuple[bool, str | None]:
    """Return (allowed, reason)."""
    low = text.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, low):
            return False, "prompt_injection"
    for name, p in BLOCKED_TOPICS.items():
        if re.search(p, low):
            return False, f"blocked_topic:{name}"
    return True, None


def redact_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, p in PII_PATTERNS.items():
        def sub(m, label=label):
            if label == "EMAIL" and m.group(0).lower() in ALLOWED_EMAILS:
                return m.group(0)
            found.append(label)
            return f"[REDACTED_{label}]"
        text = re.sub(p, sub, text)
    return text, found
