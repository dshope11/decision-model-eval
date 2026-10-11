"""Build the stage-2 (D2 + D5) sheet for a blind sample from its recorded D1 sheet.

Usage: python3 -I build_d2_sheet.py D1_SHEET DB JD_DIR OUT
WaaS JDs come from the store's jd_text; other sources from JD_DIR/L<id>.txt.
"""
# ruff: noqa: E501 - the sheet header is labeler-facing text, kept as written
import re
import sqlite3
import sys
from pathlib import Path

HEADER = """\
# sample: {sample}
# decision_point: D2+D5
# rubric_version: d2-v1
# question: D2 = the seven hard gates of fit-gate-rubric.md section 1 on the full JD; D5 = tier
# items: the {n} listings you marked D1 = y on {d1_sheet}, in sheet order
# Not read by dme label yet - this filled file is the record for D2 and D5.
#
# HOW TO FILL IT IN
#   Each JD below is the verbatim text the backends will see (WaaS: the store's jd_text;
#   others: the fetched file in jd/). Label on that text alone. If it is too thin to answer
#   from and you open the live link, add a line "opened-link" so we know.
#
#   Gates G1-G7: answer pass or fail, with ? for unsure (fail? / pass?).
#     fail = the trigger is in the REQUIRED section AND you don't clear it.
#     A trigger that only appears as preferred / nice-to-have is a pass; write pref-only after it.
#     G1 admin-stack      SLURM administration, Kubernetes operations (operators/CRDs), sysadmin duty
#     G2 clearance        an active / current TS or TS/SCI held (eligible-to-obtain is a pass)
#     G3 publication      three fields instead of pass/fail -
#                           field=  ml-venue | domain | generic | none
#                           req=    required | preferred | -        (- when field=none)
#                           clears= y | n | ?   (say how ATLAS authorship counts if it matters)
#     G4 hands-on years   production CV/perception on robots, deep RL, CUDA/Triton kernel authorship,
#                         LLM pretraining / distributed-training internals, VLM/LLM training record
#     G5 domain wall      NGS/omics, satellite-imagery preprocessing, commercial CAE solvers, quantum
#     G6 down-leveling    a junior req (L1/L2, "1+ yr") that would anchor level and comp low
#     G7 stack-miss       list the REQUIRED technologies: held= ... | not= ...  then pass/fail
#                         (>= 60% not held drops a tier; fails outright on a Senior/Staff req.
#                          No requirements section at all? the skills tags are the list.)
#   D5: very-high | high | medium | drop   (rubric section 2; any gate fail = drop)
#   D1-check: did the JD answer what your D1 note wanted to know? y | n | partly
#
# MOST USEFUL COMMENTS, in order
#   1. On every fail or ?, quote the JD line that decided it - verbatim, short, in quotes.
#   2. On a close pass, why it passes: pref-only, or wording you'd argue around.
#   3. On D5, one line on what decided it; say so if the D1 = y read was wasted.
#   4. Anything outside the rubric that moved D5 (comp, stage, team size, location detail):
#      a line starting off-rubric
#
# EXAMPLE (made up, not from this sheet)
#   [4] L1 S1   (your D1: y - "check required stack")
#       Acme Labs :: ML Engineer, Simulation :: San Francisco, CA, US
#       G1 admin-stack: pass
#       G2 clearance: pass
#       G3 publication: field=ml-venue req=preferred clears=n
#       G4 hands-on years: fail - "3+ years training large-scale RL policies (PPO, SAC) in production"
#       G5 domain wall: pass
#       G6 down-leveling: pass
#       G7 stack-miss: held=Python, PyTorch, C++ | not=Ray, Kubernetes  pass (2 of 5 not held)
#       D5: drop
#       D1-check: y
#       note: the RL line is the whole decision; everything else would have been a High
#       off-rubric: 5 people, seed stage, no comp posted
#
# Time the whole file on your stopwatch and tell Claude the minutes.
"""

ENTRY = re.compile(r"^\[(\d+)\] L(\d+) S(\d+)")


def d1_yes(sheet: Path):
    """(k, listing, snapshot, view lines, D1 note lines) for each D1 = y entry."""
    items, cur = [], None
    for raw in sheet.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        m = ENTRY.match(line)
        if m:
            cur = {"k": m.group(1), "lid": int(m.group(2)), "sid": int(m.group(3)),
                   "view": [], "notes": [], "answer": None}
            items.append(cur)
        elif cur is None or not line:
            continue
        elif line.startswith("D1:"):
            cur["answer"] = line[3:].strip().lower()
        elif cur["answer"] is None:
            cur["view"].append(line)
        else:
            cur["notes"].append(line)
    return [i for i in items if i["answer"] and i["answer"].rstrip("?") == "y"]


def main():
    sheet, db, jd_dir, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    sample = next(ln.split(":", 1)[1].strip() for ln in sheet.read_text().splitlines()
                  if ln.startswith("# sample:"))
    conn = sqlite3.connect(db)
    items = d1_yes(sheet)
    body = []
    for it in items:
        source, url = conn.execute("select source, url from listings where id = ?",
                                   (it["lid"],)).fetchone()
        if source == "waas":
            jd = conn.execute("select jd_text from snapshots where id = ?",
                              (it["sid"],)).fetchone()[0]
            where = f"store (waas)  {url}"
        else:
            path = jd_dir / f"L{it['lid']}.txt"
            text = path.read_text()
            src = re.search(r"^# source: (.*)$", text, re.M).group(1)
            fetched = re.search(r"^# fetched: (.*)$", text, re.M).group(1)
            jd = "\n".join(ln for ln in text.splitlines() if not ln.startswith("# ")).strip()
            where = f"fetched {fetched} ({source})  {src}  -> jd/{path.name}"
        d1 = f"your D1: {it['answer']}"
        body += [f"[{it['k']}] L{it['lid']} S{it['sid']}   ({d1})"]
        body += [f"    {v}" for v in it["view"]]
        body += [f"    your D1 {n}" for n in it["notes"]]
        body += [f"    JD source: {where}", "    ----- JD, verbatim -----"]
        body += [f"    | {ln}".rstrip() for ln in jd.strip().splitlines()]
        body += ["    ----- end JD -----",
                 "    G1 admin-stack:", "    G2 clearance:",
                 "    G3 publication: field= req= clears=",
                 "    G4 hands-on years:", "    G5 domain wall:", "    G6 down-leveling:",
                 "    G7 stack-miss: held= | not=", "    D5:", "    D1-check:", "    note:", "", ""]
    out.write_text(HEADER.format(sample=sample, n=len(items), d1_sheet=sheet.name)
                   + "\n" + "\n".join(body))
    print(f"{out}: {len(items)} items")


main()
