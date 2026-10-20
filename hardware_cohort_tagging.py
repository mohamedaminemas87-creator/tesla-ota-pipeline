#!/usr/bin/env python3
"""
Priority 5 - Hardware-cohort tagging (Tesla OTA pipeline)
=========================================================
Adds hardware_applicability to every classified function.

Rules (from Nicole's HARDWARE_COHORT sheet and her clarifications):
  R1  Release-date rule : branches before 2019.16 can only run on HW2.5 cars
                          (HW3 first sale = 2019-04).
  R2  Text rule         : an explicit hardware statement in the heading/body
                          overrides everything (HW3 only / AI4-HW4 only / ...).
  R3  FSD rule          : FSD >= v13 implies AI4/HW4 only; "Lite" build = HW3;
                          builds up to v12.6 cover HW3 and AI4.
  R5  Functional rule   : after HW3 launch, camera/Autopilot/FSD-dependent
                          functions are tagged HW3 (derived from the pilot
                          coding; CHECK a sample on the full corpus).
  R4  Default           : ALL (every cohort in the fleet at that date).
V01 (no front radar) and V02 (no ultrasonic) are NOT cohorts: they are
exclusion reasons INSIDE HW3 and are only reported as a flag (hw_variant_flag).

Usage:
  python hardware_cohort_tagging.py classified.xlsx cohort.xlsx out.xlsx
Evaluation on the pilot sheet:
  python hardware_cohort_tagging.py --evaluate Pilot.xlsx
"""
import re, sys
import pandas as pd

HW3_FIRST = (2019, 16)          # year, ISO-ish week of the first HW3 sales (Apr 2019)
AI4_FIRST = (2024, 1)

def branch_key(name):
    m = re.match(r"\s*(\d{4})\.(\d{1,2})", str(name))
    return (int(m.group(1)), int(m.group(2))) if m else None

def fleet_cohorts(branch):
    k = branch_key(branch)
    if k is None: return "UNKNOWN"
    c = ["HW2.5"]
    if k >= HW3_FIRST: c.append("HW3")
    if k >= AI4_FIRST: c.append("AI4")
    return "+".join(c)

RE_AI4   = re.compile(r"\b(AI4|HW4|hardware\s*4)\b", re.I)
RE_HW3   = re.compile(r"\b(HW3|hardware\s*3|FSD\s*computer)\b", re.I)
RE_HW25  = re.compile(r"\b(HW2\.5|hardware\s*2\.5|HW2)\b", re.I)
RE_LITE  = re.compile(r"\bv?\d+(\.\d+)*\s*lite\b|\blite\b", re.I)
RE_FSD   = re.compile(r"\bFSD[^.\n]{0,40}?v(\d+)(?:\.(\d+))?", re.I)
RE_FSDHEAD = re.compile(r"(?:FSD|Full Self-Driving)[^\n]{0,40}?\bv?(\d{1,2})\.(\d+)", re.I)
RE_NEG_HW3 = re.compile(r"(HW\s*3|hardware\s*3)[^.]{0,40}\b(not|no longer|except|without|cannot|n\'t)\b|\b(not|except|without)\b[^.]{0,40}(HW\s*3|hardware\s*3)", re.I)
RE_AI4ONLY = re.compile(r"(HW4|AI4|hardware\s*4(\.0)?)[^.]{0,30}\b(only|required|exclusive)|\b(only|requires?|exclusively)[^.]{0,30}(HW4|AI4|hardware\s*4)", re.I)
RE_RADAR = re.compile(r"\b(radar)\b", re.I)
RE_USS   = re.compile(r"\bultrasonic", re.I)
RE_CAMDEP= re.compile(r"\bFSD\b|full self-driving|summon|navigate on|traffic light|stop sign|cabin camera|dashcam|blind spot|driving visualiz|speed assist|pedestrian|lane change|lane keep|lane departure|autosteer|backup camera|reverse camera", re.I)
RE_VISION= re.compile(r"tesla vision|vision[- ]only|without radar", re.I)

RE_REQ_HW4 = re.compile(r"\b(HW\s*4|AI4)\b", re.I)
RE_REQ_HW3 = re.compile(r"\bHW\s*3\b", re.I)
RE_REQ_HW2 = re.compile(r"\bHW\s*2(\.5)?\s*(\+|or higher)", re.I)

def req_rule(req):
    """Hardware stated by the site in the hardware_requirement column.
    MCU / AMD Ryzen describe the infotainment computer, not the Autopilot cohort -> ignored."""
    r = "" if pd.isna(req) else str(req)
    h4, h3, h2 = bool(RE_REQ_HW4.search(r)), bool(RE_REQ_HW3.search(r)), bool(RE_REQ_HW2.search(r))
    if h4 and h3: return "HW3+AI4"
    if h4: return "AI4_ONLY"
    if h3: return "HW3"
    if h2: return "ALL"
    return None

def tag_row(branch, text, req=None, head=""):
    t = str(text)
    fleet = fleet_cohorts(branch)
    rule, app = "R4_default", "ALL"
    flag = ""
    if RE_RADAR.search(t) or RE_USS.search(t) or RE_VISION.search(t):
        flag = "V01/V02 candidate (radar/ultrasonic/Tesla Vision) - check manually"
    stated = req_rule(req)
    mh = RE_FSDHEAD.search(str(head))
    fsd_ver = (int(mh.group(1)), int(mh.group(2))) if mh else None
    k = branch_key(branch)
    ambiguous = False
    if stated:
        rule, app = "R2_stated_requirement", stated
    elif RE_LITE.search(str(head)) and fsd_ver:
        rule, app = "R3_lite", "HW3"
    elif fsd_ver and fsd_ver >= (13, 0):
        rule, app = "R3_fsd_v13+", "AI4_ONLY"
    elif fsd_ver and fsd_ver[0] >= 10 and k and k >= AI4_FIRST:
        rule, app = "R3_fsd_le12.6", "HW3+AI4"      # Nicole: builds up to v12.6 cover HW3 and AI4
    elif fsd_ver and fsd_ver[0] >= 10:
        rule, app = "R3_fsd_pre_AI4", "HW3"
    elif RE_AI4.search(t) and RE_HW3.search(t):
        rule, app, ambiguous = "R2_text", "HW3+AI4", True
    elif RE_AI4ONLY.search(t):
        rule, app, ambiguous = "R2_text", "AI4_ONLY", True
    elif RE_AI4.search(t):
        ambiguous = True            # mentions HW4 without saying only -> do not assign AI4_ONLY
        rule, app = "R4_default", "ALL"
    elif RE_HW3.search(t) and RE_NEG_HW3.search(t):
        ambiguous = True            # HW3 mentioned in a negative statement (e.g. "HW3 vehicles are not receiving ...")
        rule, app = "R4_default", "ALL"
    elif RE_HW3.search(t):
        rule, app = "R2_text", "HW3"
    elif RE_HW25.search(t):
        rule, app = "R2_text", "HW2.5"
    elif fleet == "HW2.5":
        rule, app = "R1_date", "HW2.5"
    elif RE_CAMDEP.search(str(head) + ". " + t[:300]) and "sentry" not in str(head).lower():
        rule, app = "R5_camera_dependent", ("HW3+AI4" if k and k >= AI4_FIRST else "HW3")
    return pd.Series({"hw_fleet_cohorts": fleet, "hardware_applicability_pred": app,
                      "hw_rule": rule, "hw_variant_flag": flag,
                      "hw_needs_review": ambiguous or fleet == "UNKNOWN"})

def tag(df, branch_col="branch", heading="heading", body="body_clean"):
    if body not in df.columns: body = "body"
    text = df[heading].astype(str) + ". " + df[body].astype(str)
    reqs = df["hardware_requirement"] if "hardware_requirement" in df.columns else [None] * len(df)
    out = pd.DataFrame([tag_row(b, t, q, h) for b, t, q, h in zip(df[branch_col], text, reqs, df[heading])], index=df.index)
    return pd.concat([df, out], axis=1)

def evaluate(pilot_xlsx):
    d = pd.read_excel(pilot_xlsx, sheet_name="INTERMEDIARY_FUNCTIONAL_CODING", dtype=str)
    d = d[d["function_id"].notna()].copy()
    d["text"] = d["function_name"].fillna("") + ". " + d["new_state"].fillna("")
    r = pd.DataFrame([tag_row(b, t, None, h) for b, t, h in zip(d["update_name"], d["text"], d["function_name"])], index=d.index)
    d = pd.concat([d, r], axis=1)
    gold = d["hardware_applicability"].str.strip()
    pred = d["hardware_applicability_pred"].replace({"HW3+AI4": "HW3"})   # gold HW3 = "HW3 or newer"
    print("rows:", len(d), " accuracy:", round((gold == pred).mean(), 3))
    print(pd.crosstab(gold, pred, rownames=["gold"], colnames=["pred"]))
    pre = d[d["hw_fleet_cohorts"] == "HW2.5"]
    print("\nbefore HW3 existed: rows", len(pre), " all gold HW2.5:", (pre["hardware_applicability"].str.strip() == "HW2.5").all())
    return d

if __name__ == "__main__":
    if sys.argv[1] == "--evaluate":
        evaluate(sys.argv[2])
    else:
        df = pd.read_excel(sys.argv[1], dtype={"branch": str})
        res = tag(df)
        res.to_excel(sys.argv[3], index=False)
        print("saved", sys.argv[3], "- needs_review:", int(res["hw_needs_review"].sum()))
        print(res["hardware_applicability_pred"].value_counts())
