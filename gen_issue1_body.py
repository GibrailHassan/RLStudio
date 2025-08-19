import json
import re
from pathlib import Path

ISSUES_FILE = Path('issues_m1a.json')
OUTPUT_FILE = Path('issue1_new_body.txt')

if not ISSUES_FILE.exists():
    raise SystemExit(f"Missing {ISSUES_FILE}. Run the issues export command first.")

with ISSUES_FILE.open('r', encoding='utf-8') as f:
    issues = json.load(f)

issues = sorted(issues, key=lambda x: x['number'])

def clean_title(raw: str) -> str:
    # Remove leading bracketed tags like [M1a][#54]
    cleaned = re.sub(r'^(\[[^\]]+\])+' , '', raw).strip()
    return cleaned

lines = [f"- [ ] #{it['number']} {clean_title(it['title'])}" for it in issues]

body = (
    "Umbrella tracking issue for Milestone 1a: Core Contracts & Reproducibility.\n"
    "Scope: Issues #2-#28 (initial foundation). This checklist mirrors milestone membership.\n\n"
    "Checklist:\n" + "\n".join(lines) + "\n\n"
    "Legend: [ ] open, [x] closed. Update manually (automation TBD).\n"
)

with OUTPUT_FILE.open('w', encoding='utf-8') as f:
    f.write(body)

print(f"Wrote {OUTPUT_FILE} with {len(lines)} checklist items.")
