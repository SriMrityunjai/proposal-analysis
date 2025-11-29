# ---------------------------------------------------------
# Proposal_analysis_frontend_app.py  (UPDATED)
# ---------------------------------------------------------

import streamlit as st
import requests
import json
import pandas as pd
import os
import re

# ---------------------------------------------------------
# Backend URL
# ---------------------------------------------------------
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300

st.set_page_config(page_title="RFP Proposal Analysis", layout="wide")
st.title("RFP Proposal Analysis — Streamlit")
st.info(f"Backend: {BACKEND_URL}")

# ---------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------
if "page_idx" not in st.session_state:
    st.session_state["page_idx"] = 0

if "rfp_uploaded" not in st.session_state:
    st.session_state["rfp_uploaded"] = None

if "summary" not in st.session_state:
    st.session_state["summary"] = None

if "rules" not in st.session_state:
    st.session_state["rules"] = None

if "scoring" not in st.session_state:
    st.session_state["scoring"] = None

# evaluations: list of dicts, each with keys:
# { slot:int, filename:str, requirements_df:DataFrame, scoring_df:DataFrame, raw:original_json_or_raw_text }
if "evaluations" not in st.session_state:
    st.session_state["evaluations"] = [None, None, None]

# ---------------------------------------------------------
# Page navigation
# ---------------------------------------------------------
pages = [
    "Upload RFP",
    "Evaluate Proposals",
    "Compare Evaluations"
]

def go_prev():
    st.session_state["page_idx"] = max(0, st.session_state["page_idx"] - 1)

def go_next():
    st.session_state["page_idx"] = min(len(pages) - 1, st.session_state["page_idx"] + 1)

def end_session():
    # preserve nothing: clear everything as requested
    st.session_state.clear()
    st.session_state["page_idx"] = 0
    st.session_state["evaluations"] = [None, None, None]
    st.success("Session cleared.")

st.sidebar.write(f"Page: {pages[st.session_state['page_idx']]}")
st.sidebar.button("Previous", on_click=go_prev)
st.sidebar.button("Next", on_click=go_next)
st.sidebar.button("End", on_click=end_session)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def post_file(endpoint: str, files: dict, data: dict = None):
    try:
        r = requests.post(
            f"{BACKEND_URL}{endpoint}",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

def split_two_json_arrays(raw_str: str):
    """
    Accepts either:
     - a single JSON string representing a list (in which case returns (list, []))
     - two JSON arrays back-to-back as a string (e.g. `[...] \n [...]`)
    Returns (requirements_list, scoring_list)
    """
    # try pure json loads first
    try:
        parsed = json.loads(raw_str)
        # if parsed is list and contains dicts that look like scoring vs requirements, best effort:
        if isinstance(parsed, list):
            # Heuristic: if entries have 'Point' or 'Max_Points' assume scoring; else requirements
            if parsed and any("Point" in k or "Max_Points" in k for k in parsed[0].keys()):
                return [], parsed
            else:
                return parsed, []
        else:
            return [], []
    except Exception:
        pass

    # fallback: find two array blocks using regex
    m = re.findall(r"(\[.*?\])", raw_str, flags=re.S)
    if len(m) >= 2:
        try:
            first = json.loads(m[0])
        except Exception:
            first = []
        try:
            second = json.loads(m[1])
        except Exception:
            second = []
        # decide which is which by heuristics
        reqs = first
        scoring = second
        # if second looks requirements-like, swap
        def looks_requirements(lst):
            if not isinstance(lst, list) or not lst:
                return False
            keys = lst[0].keys()
            return any(k.lower().startswith("rfp") or "address" in k.lower() for k in keys)
        if looks_requirements(second) and not looks_requirements(first):
            reqs, scoring = second, first
        return reqs, scoring

    # nothing parsed
    return [], []

def evaluation_to_tables(raw_eval):
    """
    raw_eval may be:
     - dict with key 'evaluations' : [ ... ] (older backend)
     - string with two arrays back-to-back
     - dict with 'requirements' and 'scoring' keys (possible)
    Return (req_df, score_df)
    """
    # If it's already a dict with lists:
    if isinstance(raw_eval, dict):
        # If has 'evaluations' as list of per-item (older format) try to split
        if "evaluations" in raw_eval:
            # try to detect schema: if items have 'RFP Requirement/Criteria' or 'Requirement'
            evs = raw_eval["evaluations"]
            # If these appear to be requirements or scoring? We will attempt to split into two tables:
            # Heuristic split by presence of Score or Max Points
            req_rows = []
            score_rows = []
            for ev in evs:
                if any(k.lower() in ("point","max_points","score","max points","score_awarded") for k in ev.keys()):
                    # scoring-like
                    score_rows.append(ev)
                else:
                    req_rows.append(ev)
            req_df = pd.json_normalize(req_rows) if req_rows else pd.DataFrame()
            score_df = pd.json_normalize(score_rows) if score_rows else pd.DataFrame()
            return req_df, score_df

        # If keys are explicit:
        if "requirements" in raw_eval and "scoring" in raw_eval:
            try:
                req_df = pd.DataFrame(raw_eval["requirements"])
            except Exception:
                req_df = pd.DataFrame()
            try:
                score_df = pd.DataFrame(raw_eval["scoring"])
            except Exception:
                score_df = pd.DataFrame()
            return req_df, score_df

    # If string:
    if isinstance(raw_eval, str):
        req_list, scoring_list = split_two_json_arrays(raw_eval)
        req_df = pd.DataFrame(req_list) if req_list else pd.DataFrame()
        score_df = pd.DataFrame(scoring_list) if scoring_list else pd.DataFrame()
        return req_df, score_df

    # unknown type
    return pd.DataFrame(), pd.DataFrame()

# ---------------------------------------------------------
# PAGE 1 — Upload RFP
# ---------------------------------------------------------
page = pages[st.session_state["page_idx"]]

if page == "Upload RFP":
    st.header("1 — Upload RFP (Summary + Rules + Scoring)")

    uploaded = st.file_uploader("Upload RFP (pdf / docx / txt)", type=["pdf", "docx", "txt"])

    if uploaded:
        st.session_state["rfp_uploaded"] = uploaded.name
        st.info("Processing RFP…")

        files = {"file": (uploaded.name, uploaded.getvalue())}
        resp = post_file("/extract_rfp", files)

        if resp and "error" not in resp:
            st.session_state["summary"] = resp.get("summary", "")
            st.session_state["rules"] = resp.get("rules", [])
            st.session_state["scoring"] = resp.get("scoring", [])

            # ---------------- Summary ----------------
            st.subheader("Summary (max 300 words)")
            df_summary = pd.DataFrame([{"Summary": st.session_state["summary"]}])
            st.table(df_summary)

            st.download_button(
                "Download Summary CSV",
                df_summary.to_csv(index=False),
                file_name="summary.csv",
            )

            # ---------------- Rules ----------------
            st.subheader("Rules — Table Format")
            df_rules = pd.DataFrame(st.session_state["rules"])
            st.dataframe(df_rules)

            st.download_button(
                "Download Rules CSV",
                df_rules.to_csv(index=False),
                file_name="rules.csv",
            )

            # ---------------- Scoring ----------------
            st.subheader("Scoring — Table Format")
            df_scoring = pd.DataFrame(st.session_state["scoring"])
            st.dataframe(df_scoring)

            st.download_button(
                "Download Scoring CSV",
                df_scoring.to_csv(index=False),
                file_name="scoring.csv",
            )

        elif resp and "error" in resp:
            st.error(resp["error"])
        else:
            st.error("No response from backend.")

# ---------------------------------------------------------
# PAGE 2 — Evaluate Proposals
# ---------------------------------------------------------
elif page == "Evaluate Proposals":
    st.header("2 — Upload Proposal(s) for Evaluation (max 3)")

    cols = st.columns(3)

    for idx in range(3):
        with cols[idx]:
            st.markdown(f"### Slot {idx+1}")

            uploaded = st.file_uploader(
                f"Slot {idx+1}: Upload Proposal or Evaluation JSON",
                key=f"file_{idx}",
                type=["pdf", "docx", "txt", "json"],
            )

            evaluate_btn = st.button(f"Evaluate Slot {idx+1}", key=f"eval_{idx}")

            if uploaded:
                if uploaded.name.endswith(".json"):  # existing evaluation JSON
                    try:
                        result = json.loads(uploaded.getvalue().decode())
                        req_df, score_df = evaluation_to_tables(result)
                        st.session_state["evaluations"][idx] = {
                            "slot": idx + 1,
                            "filename": uploaded.name,
                            "requirements_df": req_df,
                            "scoring_df": score_df,
                            "raw": result,
                        }
                        st.success(f"Loaded evaluation JSON → Slot {idx+1}")
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")

                else:  # proposal file
                    st.write(f"Uploaded: **{uploaded.name}**")

                    if evaluate_btn:
                        if not st.session_state["rules"] or not st.session_state["scoring"]:
                            st.error("Upload RFP on Page 1 to extract rules + scoring first.")
                        else:
                            files = {"file": (uploaded.name, uploaded.getvalue())}
                            data = {
                                "rules_json": json.dumps(st.session_state["rules"]),
                                "scoring_json": json.dumps(st.session_state["scoring"]),
                            }
                            st.info("Evaluating proposal…")
                            resp = post_file("/evaluate_proposal", files, data)
                            resp = post_file("/evaluate_proposal", files, data)

# ---------- SAFE ERROR HANDLING FIX ----------
                            if resp is None:
                                st.error("Backend returned no response — evaluation failed.")
                            elif "error" in resp:
                                st.error(resp["error"])
                            else:
                                raw_result = resp

                                if isinstance(resp, dict) and "evaluations" in resp:
                                    raw_payload = resp["evaluations"]
                                else:
                                    raw_payload = resp

                                req_df, score_df = evaluation_to_tables(
                                    raw_payload if not isinstance(raw_payload, dict) else raw_payload
                                )

                                st.session_state["evaluations"][idx] = {
                                    "slot": idx + 1,
                                    "filename": uploaded.name,
                                    "requirements_df": req_df,
                                    "scoring_df": score_df,
                                    "raw": resp,
                                }
                                st.success(f"Evaluation Completed → Slot {idx+1}")
# ----------------------------------------------


                            if resp and "error" not in resp:
                                # resp may be {"evaluations": [...] } or raw string arrays depending on backend
                                # Accept both dict and string
                                raw_result = resp
                                # If resp contains 'evaluations' with raw text, attempt to extract
                                if isinstance(resp, dict) and "evaluations" in resp:
                                    raw_payload = resp["evaluations"]
                                else:
                                    raw_payload = resp

                                # Convert to tables
                                req_df, score_df = evaluation_to_tables(raw_payload if not isinstance(raw_payload, dict) else raw_payload)
                                st.session_state["evaluations"][idx] = {
                                    "slot": idx + 1,
                                    "filename": uploaded.name,
                                    "requirements_df": req_df,
                                    "scoring_df": score_df,
                                    "raw": resp,
                                }
                                st.success(f"Evaluation Completed → Slot {idx+1}")
                            else:
                                st.error(resp.get("error", "Unknown evaluation error"))

            # Show slot content if available
            slot_data = st.session_state["evaluations"][idx]
            if slot_data:
                st.write(f"**{slot_data['filename']} — Slot {idx+1}**")
                # Requirements table
                req_df = slot_data.get("requirements_df", pd.DataFrame())
                if not req_df.empty:
                    st.subheader("Requirements / Rules (table)")
                    st.dataframe(req_df)
                    st.download_button(
                        f"Download Slot {idx+1} Requirements CSV",
                        req_df.to_csv(index=False),
                        file_name=f"slot{idx+1}_requirements.csv",
                    )
                else:
                    st.info("No requirements table produced for this evaluation.")

                # Scoring table
                score_df = slot_data.get("scoring_df", pd.DataFrame())
                if not score_df.empty:
                    st.subheader("Scoring (table)")
                    st.dataframe(score_df)
                    st.download_button(
                        f"Download Slot {idx+1} Scoring CSV",
                        score_df.to_csv(index=False),
                        file_name=f"slot{idx+1}_scoring.csv",
                    )
                else:
                    st.info("No scoring table produced for this evaluation.")

# ---------------------------------------------------------
# PAGE 3 — Compare Evaluations (local comparison)
# ---------------------------------------------------------
elif page == "Compare Evaluations":
    st.header("3 — Compare up to 3 Evaluation Reports")

    available = [e for e in st.session_state["evaluations"] if e]

    if not available:
        st.info("No evaluations found. Upload proposals on Page 2.")
    else:
        st.subheader("Loaded Evaluations")
        for e in available:
            st.write(f"- {e['filename']} (Slot {e['slot']})")

        if st.button("Compute Comparison"):
            # Build rules comparison table (rows = union of all requirements)
            all_reqs = {}
            for e in available:
                df = e.get("requirements_df", pd.DataFrame())
                if df.empty:
                    continue
                # try to extract a canonical requirement name column
                name_col = None
                for c in df.columns:
                    if "require" in c.lower() or "criteria" in c.lower() or "rfp" in c.lower():
                        name_col = c
                        break
                if not name_col:
                    # fallback to first column
                    name_col = df.columns[0] if len(df.columns) > 0 else None
                for _, row in df.iterrows():
                    req_name = str(row[name_col]) if name_col else "Unknown Requirement"
                    addr = None
                    for c in df.columns:
                        if "address" in c.lower():
                            addr = row[c]
                            break
                    addr_val = "Yes" if str(addr).strip().lower() in ("yes", "true", "y", "1") else "No"
                    if req_name not in all_reqs:
                        all_reqs[req_name] = {}
                    all_reqs[req_name][e["filename"]] = addr_val

            rules_comp_df = pd.DataFrame.from_dict(all_reqs, orient="index").fillna("No")
            st.subheader("Rules Comparison")
            st.dataframe(rules_comp_df)
            st.download_button("Download Rules Comparison CSV", rules_comp_df.to_csv(index=True), file_name="rules_comparison.csv")

            # Build scoring comparison (per scoring point rows plus totals)
            score_points = {}
            totals = {}
            for e in available:
                df = e.get("scoring_df", pd.DataFrame())
                if df.empty:
                    totals[e["filename"]] = 0
                    continue
                # find point name column
                point_col = None
                score_col = None
                for c in df.columns:
                    if "point" in c.lower() or "metric" in c.lower():
                        point_col = c
                    if "score" in c.lower() or "score_awarded" in c.lower() or "score awarded" in c.lower():
                        score_col = c
                if not point_col and df.shape[1] >= 1:
                    point_col = df.columns[0]
                if not score_col and df.shape[1] >= 2:
                    score_col = df.columns[1]

                total = 0
                for _, row in df.iterrows():
                    point_name = str(row[point_col]) if point_col else "Point"
                    try:
                        val = float(row[score_col]) if score_col and row[score_col] not in (None, "") else 0.0
                    except Exception:
                        # try parse numbers inside string
                        try:
                            val = float(re.findall(r"[-+]?\d*\.\d+|\d+", str(row[score_col]))[0])
                        except Exception:
                            val = 0.0
                    if point_name not in score_points:
                        score_points[point_name] = {}
                    score_points[point_name][e["filename"]] = val
                    total += val
                totals[e["filename"]] = total

            score_comp_df = pd.DataFrame.from_dict(score_points, orient="index").fillna(0)
            # append totals row
            totals_row = pd.DataFrame(totals, index=["Total"])
            score_comp_with_totals = pd.concat([score_comp_df, totals_row])
            st.subheader("Score Comparison (points per metric + Total)")
            st.dataframe(score_comp_with_totals)
            st.download_button("Download Score Comparison CSV", score_comp_with_totals.to_csv(index=True), file_name="score_comparison.csv")

            # Simple ranking by totals
            ranking = sorted(totals.items(), key=lambda x: x[1], reverse=True)
            st.subheader("Ranking by Total Score")
            for i, (name, val) in enumerate(ranking, start=1):
                st.write(f"{i}. {name} — Total points: {val}")

        # Optionally allow backend compare for LLM reasoning (still returns JSON which we'll show as text)
        if st.button("Ask backend LLM for reasoning (optional)"):
            files_payload = []
            for e in available:
                content = json.dumps({
                    "evaluations": []
                })
                # prefer to send the original raw if available
                raw = e.get("raw")
                if raw:
                    # if raw is dict, send that. If DataFrames exist, create combined structure
                    try:
                        if isinstance(raw, (dict, list)):
                            content = json.dumps(raw)
                        else:
                            # fallback: create small structured payload
                            content = json.dumps({
                                "requirements": json.loads(e["requirements_df"].to_json(orient="records")) if not e["requirements_df"].empty else [],
                                "scoring": json.loads(e["scoring_df"].to_json(orient="records")) if not e["scoring_df"].empty else []
                            })
                    except Exception:
                        content = json.dumps({
                            "requirements": json.loads(e["requirements_df"].to_json(orient="records")) if not e["requirements_df"].empty else [],
                            "scoring": json.loads(e["scoring_df"].to_json(orient="records")) if not e["scoring_df"].empty else []
                        })
                files_payload.append(("files", (e["filename"], content, "application/json")))

            resp = post_file("/compare_reports", files=files_payload)
            if resp and "error" not in resp:
                st.subheader("Backend LLM Recommendation / Analysis (JSON)")
                st.json(resp.get("recommendation", resp))
            else:
                st.error("Backend compare failed.")



# # ---------------------------------------------------------
# # Proposal_analysis_frontend_app.py  (FULL WORKING VERSION)
# # ---------------------------------------------------------

# import streamlit as st
# import requests
# import json
# import pandas as pd
# import os

# # ---------------------------------------------------------
# # Backend URL
# # ---------------------------------------------------------
# BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
# REQUEST_TIMEOUT = 300

# st.set_page_config(page_title="RFP Proposal Analysis", layout="wide")
# st.title("RFP Proposal Analysis — Streamlit")
# st.info(f"Backend: {BACKEND_URL}")

# # ---------------------------------------------------------
# # Session state defaults
# # ---------------------------------------------------------
# if "page_idx" not in st.session_state:
#     st.session_state["page_idx"] = 0

# if "rfp_uploaded" not in st.session_state:
#     st.session_state["rfp_uploaded"] = None

# if "summary" not in st.session_state:
#     st.session_state["summary"] = None

# if "rules" not in st.session_state:
#     st.session_state["rules"] = None

# if "scoring" not in st.session_state:
#     st.session_state["scoring"] = None

# if "evaluations" not in st.session_state:
#     st.session_state["evaluations"] = [None, None, None]

# # ---------------------------------------------------------
# # Page navigation
# # ---------------------------------------------------------
# pages = [
#     "Upload RFP",
#     "Evaluate Proposals",
#     "Compare Evaluations"
# ]

# def go_prev():
#     st.session_state["page_idx"] = max(0, st.session_state["page_idx"] - 1)

# def go_next():
#     st.session_state["page_idx"] = min(len(pages) - 1, st.session_state["page_idx"] + 1)

# def end_session():
#     st.session_state.clear()
#     st.session_state["page_idx"] = 0
#     st.session_state["evaluations"] = [None, None, None]
#     st.success("Session cleared.")

# st.sidebar.write(f"Page: {pages[st.session_state['page_idx']]}")
# st.sidebar.button("Previous", on_click=go_prev)
# st.sidebar.button("Next", on_click=go_next)
# st.sidebar.button("End", on_click=end_session)

# # ---------------------------------------------------------
# # Helper functions
# # ---------------------------------------------------------
# def post_file(endpoint: str, files: dict, data: dict = None):
#     try:
#         r = requests.post(
#             f"{BACKEND_URL}{endpoint}",
#             files=files,
#             data=data,
#             timeout=REQUEST_TIMEOUT
#         )
#         r.raise_for_status()
#         return r.json()
#     except Exception as e:
#         st.error(f"Request failed: {e}")
#         return None


# # ---------------------------------------------------------
# # PAGE 1 — Upload RFP
# # ---------------------------------------------------------
# page = pages[st.session_state["page_idx"]]

# if page == "Upload RFP":
#     st.header("1 — Upload RFP (Summary + Rules + Scoring)")

#     uploaded = st.file_uploader("Upload RFP (pdf / docx / txt)", type=["pdf", "docx", "txt"])

#     if uploaded:
#         st.session_state["rfp_uploaded"] = uploaded.name
#         st.info("Processing RFP…")

#         files = {"file": (uploaded.name, uploaded.getvalue())}
#         resp = post_file("/extract_rfp", files)

#         if resp and "error" not in resp:
#             st.session_state["summary"] = resp.get("summary", "")
#             st.session_state["rules"] = resp.get("rules", [])
#             st.session_state["scoring"] = resp.get("scoring", [])

#             # ---------------- Summary ----------------
#             st.subheader("Summary (max 300 words)")
#             df_summary = pd.DataFrame([{"Summary": st.session_state["summary"]}])
#             st.table(df_summary)

#             st.download_button(
#                 "Download Summary CSV",
#                 df_summary.to_csv(index=False),
#                 file_name="summary.csv",
#             )

#             # ---------------- Rules ----------------
#             st.subheader("Rules — Table Format")
#             df_rules = pd.DataFrame(st.session_state["rules"])
#             st.dataframe(df_rules)

#             st.download_button(
#                 "Download Rules CSV",
#                 df_rules.to_csv(index=False),
#                 file_name="rules.csv",
#             )

#             # ---------------- Scoring ----------------
#             st.subheader("Scoring — Table Format")
#             df_scoring = pd.DataFrame(st.session_state["scoring"])
#             st.dataframe(df_scoring)

#             st.download_button(
#                 "Download Scoring CSV",
#                 df_scoring.to_csv(index=False),
#                 file_name="scoring.csv",
#             )

#         elif resp and "error" in resp:
#             st.error(resp["error"])
#         else:
#             st.error("No response from backend.")


# # ---------------------------------------------------------
# # PAGE 2 — Evaluate Proposals
# # ---------------------------------------------------------
# elif page == "Evaluate Proposals":
#     st.header("2 — Upload Proposal(s) for Evaluation")

#     cols = st.columns(3)

#     for idx in range(3):
#         with cols[idx]:
#             st.markdown(f"### Slot {idx+1}")

#             uploaded = st.file_uploader(
#                 f"Slot {idx+1}: Upload Proposal or Evaluation JSON",
#                 key=f"file_{idx}",
#                 type=["pdf", "docx", "txt", "json"],
#             )

#             evaluate_btn = st.button(f"Evaluate Slot {idx+1}", key=f"eval_{idx}")

#             if uploaded:
#                 if uploaded.name.endswith(".json"):  # existing evaluation
#                     try:
#                         result = json.loads(uploaded.getvalue().decode())
#                         st.session_state["evaluations"][idx] = {
#                             "slot": idx + 1,
#                             "filename": uploaded.name,
#                             "result": result,
#                         }
#                         st.success(f"Loaded evaluation JSON → Slot {idx+1}")
#                     except Exception as e:
#                         st.error(f"Invalid JSON: {e}")

#                 else:  # proposal file
#                     st.write(f"Uploaded: **{uploaded.name}**")

#                     if evaluate_btn:
#                         if not st.session_state["rules"] or not st.session_state["scoring"]:
#                             st.error("Upload RFP on Page 1 to extract rules + scoring first.")
#                         else:
#                             files = {"file": (uploaded.name, uploaded.getvalue())}
#                             data = {
#                                 "rules_json": json.dumps(st.session_state["rules"]),
#                                 "scoring_json": json.dumps(st.session_state["scoring"]),
#                             }
#                             st.info("Evaluating proposal…")
#                             resp = post_file("/evaluate_proposal", files, data)

#                             if resp and "error" not in resp:
#                                 st.session_state["evaluations"][idx] = {
#                                     "slot": idx + 1,
#                                     "filename": uploaded.name,
#                                     "result": resp,
#                                 }
#                                 st.success(f"Evaluation Completed → Slot {idx+1}")
#                             else:
#                                 st.error(resp.get("error", "Unknown evaluation error"))

#             # Show slot content
#             slot_data = st.session_state["evaluations"][idx]
#             if slot_data:
#                 st.json(slot_data["result"])
#                 st.download_button(
#                     f"Download Slot {idx+1} Evaluation JSON",
#                     json.dumps(slot_data["result"], indent=2),
#                     file_name=f"slot{idx+1}_evaluation.json",
#                 )


# # ---------------------------------------------------------
# # PAGE 3 — Compare Evaluations
# # ---------------------------------------------------------
# elif page == "Compare Evaluations":
#     st.header("3 — Compare up to 3 Evaluation Reports")

#     available = [e for e in st.session_state["evaluations"] if e]

#     if not available:
#         st.info("No evaluations found. Upload proposals on Page 2.")
#     else:
#         st.subheader("Loaded Evaluations")
#         for e in available:
#             st.write(f"- {e['filename']}")

#         if st.button("Compare Now"):
#             files_payload = []
#             for e in available:
#                 content = json.dumps(e["result"])
#                 files_payload.append(
#                     ("files", (e["filename"], content, "application/json"))
#                 )

#             resp = post_file("/compare_reports", files=files_payload)

#             if resp and "error" not in resp:
#                 st.subheader("Comparison Result")
#                 st.json(resp)
#             else:
#                 st.error(resp.get("error", "Comparison failed"))
# ######################################################################


# # # # Proposal_analysis_frontend_app.py

# # # -----------------------------
# # # Page 1: Upload RFP
# # # -----------------------------
# # page = pages[st.session_state['page_idx']]
# # if page == "Upload RFP":
# #     st.header("1 — Upload RFP (Summary, Rules, Scoring)")

# #     uploaded = st.file_uploader("Upload RFP (pdf/docx/txt)", type=["pdf","docx","txt"])

# #     if uploaded:
# #         st.session_state['rfp_uploaded'] = uploaded.name
# #         st.info("Processing RFP…")

# #         files = {"file": (uploaded.name, uploaded.getvalue())}
# #         resp = post_file("/extract_rfp", files)

# #         if resp is None:
# #             st.error("No response from backend.")
# #         elif "error" in resp:
# #             st.error(resp["error"])
# #         else:
# #             st.session_state['summary'] = resp.get("summary", "")
# #             st.session_state['rules'] = resp.get("rules", [])
# #             st.session_state['scoring'] = resp.get("scoring", [])

# #             # --- SUMMARY TABLE ---
# #             st.subheader("Summary")
# #             df_summary = pd.DataFrame([{"Summary": st.session_state['summary']}])
# #             st.table(df_summary)
# #             st.download_button("Download Summary CSV",
# #                                df_summary.to_csv(index=False),
# #                                file_name="summary.csv")

# #             # --- RULES TABLE ---
# #             st.subheader("Rules")
# #             df_rules = pd.DataFrame(st.session_state['rules'])
# #             st.dataframe(df_rules)
# #             st.download_button("Download Rules CSV",
# #                                df_rules.to_csv(index=False),
# #                                file_name="rules.csv")

# #             # --- SCORING TABLE ---
# #             st.subheader("Scoring")
# #             df_scoring = pd.DataFrame(st.session_state['scoring'])
# #             st.dataframe(df_scoring)
# #             st.download_button("Download Scoring CSV",
# #                                df_scoring.to_csv(index=False),
# #                                file_name="scoring.csv")



# # # # Proposal_analysis_frontend_app.py
# # # import streamlit as st
# # # import requests
# # # import json
# # # import pandas as pd
# # # import os

# # # BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
# # # #BACKEND_URL = os.environ.get("BACKEND_URL", "https://cf532fa9de1f.ngrok-free.app")

# # # REQUEST_TIMEOUT = 300  # seconds (match backend)
# # # st.set_page_config(page_title="RFP Proposal Analysis POC", layout="wide")
# # # st.title("RFP Proposal Analysis — Streamlit POC")
# # # st.info(f"Backend: {BACKEND_URL}")

# # # # -----------------------------
# # # # Session state initialization
# # # # -----------------------------
# # # if 'rfp_uploaded' not in st.session_state:
# # #     st.session_state['rfp_uploaded'] = None
# # # if 'summary' not in st.session_state:
# # #     st.session_state['summary'] = None
# # # if 'rules' not in st.session_state:
# # #     st.session_state['rules'] = None
# # # if 'scoring' not in st.session_state:
# # #     st.session_state['scoring'] = None
# # # # evaluations: list of dicts {"slot": 1/2/3, "filename": name, "result": {...}}
# # # if 'evaluations' not in st.session_state:
# # #     st.session_state['evaluations'] = [None, None, None]  # indices 0,1,2 for slots 1..3
# # # if 'page_idx' not in st.session_state:
# # #     st.session_state['page_idx'] = 0

# # # # -----------------------------
# # # # Navigation
# # # # -----------------------------
# # # pages = ["Upload RFP", "Evaluate / Upload Reports", "Compare Evaluations"]

# # # def go_prev():
# # #     st.session_state['page_idx'] = max(0, st.session_state['page_idx'] - 1)

# # # def go_next():
# # #     st.session_state['page_idx'] = min(len(pages)-1, st.session_state['page_idx'] + 1)

# # # def end_session():
# # #     # keep RFP summary/rules/scoring visible on page1 after end? user requested clearing until End — End clears everything
# # #     st.session_state.clear()
# # #     st.session_state['evaluations'] = [None, None, None]
# # #     st.session_state['page_idx'] = 0
# # #     st.success("Session cleared.")

# # # st.sidebar.write(f"Page: {pages[st.session_state['page_idx']]}")
# # # st.sidebar.button("Previous", on_click=go_prev)
# # # st.sidebar.button("Next", on_click=go_next)
# # # st.sidebar.button("End", on_click=end_session)

# # # # -----------------------------
# # # # Helpers
# # # # -----------------------------
# # # def post_file(endpoint: str, files: dict, data: dict = None):
# # #     url = f"{BACKEND_URL}{endpoint}"
# # #     try:
# # #         r = requests.post(url, files=files, data=data, timeout=REQUEST_TIMEOUT)
# # #         r.raise_for_status()
# # #         return r.json()
# # #     except requests.exceptions.HTTPError:
# # #         try:
# # #             return r.json()
# # #         except Exception as e:
# # #             st.error(f"Backend HTTP error: {e}")
# # #             return None
# # #     except Exception as e:
# # #         st.error(f"Request failed: {e}")
# # #         return None

# # # def post_compare(files_list):
# # #     url = f"{BACKEND_URL}/compare_reports"
# # #     try:
# # #         r = requests.post(url, files=files_list, timeout=REQUEST_TIMEOUT)
# # #         r.raise_for_status()
# # #         return r.json()
# # #     except Exception as e:
# # #         st.error(f"Compare request failed: {e}")
# # #         return None

# # # # -----------------------------
# # # # Page 1: Upload RFP
# # # # -----------------------------
# # # page = pages[st.session_state['page_idx']]
# # # if page == "Upload RFP":
# # #     st.header("1 — Upload RFP document (summary, rules, scoring)")
# # #     uploaded = st.file_uploader("Upload RFP (pdf/docx/txt)", type=["pdf","docx","txt"])
# # #     if uploaded:
# # #         st.session_state['rfp_uploaded'] = uploaded.name
# # #         st.info("Sending RFP for extraction (summary, rules, scoring)...")
# # #         files = {"file": (uploaded.name, uploaded.getvalue())}
# # #         resp = post_file("/extract_rfp", files)
# # #         if resp is None:
# # #             st.error("No response from backend.")
# # #         else:
# # #             if "error" in resp:
# # #                 st.error(resp["error"])
# # #             else:
# # #                 st.session_state['summary'] = resp.get("summary", "")
# # #                 st.session_state['rules'] = resp.get("rules", [])
# # #                 st.session_state['scoring'] = resp.get("scoring", [])
# # #                 # show results
# # #                 st.subheader("Summary")
# # #                 st.write(st.session_state['summary'] or "(no summary)")
# # #                 st.subheader("Extracted Rules")
# # #                 st.json(st.session_state['rules'] or [])
# # #                 st.subheader("Scoring")
# # #                 st.json(st.session_state['scoring'] or [])
# # #                 if resp.get("errors"):
# # #                     st.warning("Partial extraction issues:")
# # #                     for e in resp.get("errors", []):
# # #                         st.write("-", e)
# # #                 st.download_button("Download rules JSON", json.dumps(st.session_state['rules'], indent=2),
# # #                                    file_name="rfp_rules.json")
# # #                 st.download_button("Download scoring JSON", json.dumps(st.session_state['scoring'], indent=2),
# # #                                    file_name="rfp_scoring.json")
# # #     else:
# # #         if st.session_state['summary']:
# # #             st.subheader("Summary (from session)")
# # #             st.write(st.session_state['summary'])
# # #             st.subheader("Extracted Rules (from session)")
# # #             st.json(st.session_state['rules'] or [])
# # #             st.subheader("Scoring (from session)")
# # #             st.json(st.session_state['scoring'] or [])

# # # # -----------------------------
# # # # Page 2: Evaluate / Upload Reports (three slots)
# # # # -----------------------------
# # # elif page == "Evaluate / Upload Reports":
# # #     st.header("2 — Drag or upload evaluation/proposal files (3 slots)")
# # #     st.write("For each slot you can either upload a proposal (pdf/docx/txt) and click Evaluate, or upload an existing evaluation JSON file.")
# # #     cols = st.columns(3)
# # #     for idx in range(3):
# # #         with cols[idx]:
# # #             st.markdown(f"**Slot {idx+1}**")
# # #             st.write("Drag your file:")
# # #             # allow JSON (existing evaluation) or proposal file
# # #             uploaded_slot = st.file_uploader(label=f"Slot {idx+1} — Proposal or evaluation JSON", key=f"slot_{idx+1}", type=["pdf","docx","txt","json"], accept_multiple_files=False)
# # #             eval_btn = st.button(f"Evaluate Slot {idx+1} (if proposal uploaded)", key=f"eval_btn_{idx+1}")
# # #             # Handle upload
# # #             if uploaded_slot:
# # #                 # if JSON — treat as evaluation result
# # #                 if uploaded_slot.type == "application/json" or uploaded_slot.name.lower().endswith(".json"):
# # #                     try:
# # #                         content = json.loads(uploaded_slot.getvalue().decode())
# # #                         st.session_state['evaluations'][idx] = {"slot": idx+1, "filename": uploaded_slot.name, "result": content}
# # #                         st.success(f"Loaded evaluation JSON into slot {idx+1}")
# # #                     except Exception as e:
# # #                         st.error(f"Failed to parse JSON: {e}")
# # #                 else:
# # #                     # proposal file: show name, allow Evaluate
# # #                     st.write(f"Uploaded proposal: **{uploaded_slot.name}**")
# # #                     if eval_btn:
# # #                         # need rules & scoring available
# # #                         if not st.session_state.get('rules') or not st.session_state.get('scoring'):
# # #                             st.error("Please upload RFP on Page 1 to extract rules & scoring first.")
# # #                         else:
# # #                             st.info(f"Evaluating {uploaded_slot.name} (slot {idx+1})...")
# # #                             files = {"file": (uploaded_slot.name, uploaded_slot.getvalue())}
# # #                             data = {"rules_json": json.dumps(st.session_state['rules']), "scoring_json": json.dumps(st.session_state['scoring'])}
# # #                             resp = post_file("/evaluate_proposal", files, data)
# # #                             if resp is None:
# # #                                 st.error("No response from backend.")
# # #                             else:
# # #                                 if "error" in resp:
# # #                                     st.error(f"Evaluation error: {resp.get('error')}")
# # #                                 else:
# # #                                     st.session_state['evaluations'][idx] = {"slot": idx+1, "filename": uploaded_slot.name, "result": resp}
# # #                                     st.success(f"Evaluation stored in slot {idx+1}")
# # #             # show current slot evaluation (if exists)
# # #             cur = st.session_state['evaluations'][idx]
# # #             if cur:
# # #                 st.markdown("**Evaluation in this slot:**")
# # #                 st.write(f"File: {cur.get('filename')}")
# # #                 st.json(cur.get('result'))
# # #                 # download buttons for convenience
# # #                 st.download_button(f"Download Slot{idx+1} evaluation JSON", json.dumps(cur.get('result'), indent=2), file_name=f"slot{idx+1}_evaluation.json")
# # #                 csv_df = pd.json_normalize(cur.get('result', {}).get('evaluations', []))
# # #                 if not csv_df.empty:
# # #                     st.download_button(f"Download Slot{idx+1} CSV", csv_df.to_csv(index=False), file_name=f"slot{idx+1}_evaluation.csv")

# # # # -----------------------------
# # # # Page 3: Compare Evaluations
# # # # -----------------------------
# # # elif page == "Compare Evaluations":
# # #     st.header("3 — Compare evaluation reports and produce final ranking (Style A)")
# # #     # prepare list of available slots
# # #     available = []
# # #     for i, e in enumerate(st.session_state['evaluations']):
# # #         if e:
# # #             available.append((i+1, e['filename']))
# # #     if not available:
# # #         st.info("No evaluation reports in session. Upload/evaluate on Page 2 first or upload JSON here.")
# # #     # allow upload of additional evaluation JSONs (optional)
# # #     uploaded_jsons = st.file_uploader("Or upload evaluation JSON files here (optional)", accept_multiple_files=True, type=['json'])
# # #     combined = []
# # #     # add session ones first
# # #     for i, e in enumerate(st.session_state['evaluations']):
# # #         if e:
# # #             combined.append({"slot": i+1, "filename": e['filename'], "result": e['result']})
# # #     # add uploaded_jsons
# # #     if uploaded_jsons:
# # #         for uj in uploaded_jsons:
# # #             try:
# # #                 combined.append({"slot": None, "filename": uj.name, "result": json.loads(uj.getvalue().decode())})
# # #             except Exception as ex:
# # #                 st.error(f"Failed to parse {uj.name}: {ex}")

# # #     if combined:
# # #         st.subheader("Available evaluation reports")
# # #         for i, c in enumerate(combined):
# # #             st.write(f"{i+1}. {c['filename']}")
# # #         # selection: pick 1..3 reports to compare (by index in combined)
# # #         opts = [f"{i+1}: {c['filename']}" for i, c in enumerate(combined)]
# # #         chosen = st.multiselect("Select reports to compare (choose 1, 2 or 3)", opts, default=opts[:min(3,len(opts))])
# # #         if chosen:
# # #             # build files payload for backend compare endpoint
# # #             files_payload = []
# # #             # map chosen strings back to combined indices
# # #             for sel in chosen:
# # #                 idx = int(sel.split(":")[0]) - 1
# # #                 entry = combined[idx]
# # #                 files_payload.append(("files", (entry['filename'], json.dumps(entry['result']).encode('utf-8'), "application/json")))
# # #             if st.button("Compare selected reports (ask LLM)"):
# # #                 with st.spinner("Comparing and asking model for ranking..."):
# # #                     resp = post_compare(files_payload)
# # #                     if resp:
# # #                         if "error" in resp:
# # #                             st.error(resp["error"])
# # #                         else:
# # #                             st.subheader("Comparison Table")
# # #                             # show dataframe from csv if possible
# # #                             try:
# # #                                 df = pd.read_csv(pd.compat.StringIO(resp.get("csv")))
# # #                             except Exception:
# # #                                 # fallback: build from comparison_table
# # #                                 df = pd.DataFrame.from_dict(resp.get("comparison_table"), orient='index').fillna('')
# # #                             st.dataframe(df)
# # #                             st.download_button("Download comparison CSV", resp.get("csv", df.to_csv(index=True)))
# # #                             st.subheader("Ranking (Style A)")
# # #                             ranking = resp.get("ranking", [])
# # #                             if ranking:
# # #                                 for i, r in enumerate(ranking, start=1):
# # #                                     pts = r.get("total_points")
# # #                                     if pts is None:
# # #                                         pts_display = "N/A"
# # #                                     else:
# # #                                         # show as integer if whole
# # #                                         pts_display = int(pts) if float(pts).is_integer() else float(pts)
# # #                                     st.write(f"{i}. {r.get('filename')} — {pts_display} points")
# # #                             st.subheader("Recommendation / Analysis")
# # #                             st.json(resp.get("recommendation", {}))
# # #                             st.download_button("Download recommendation JSON", json.dumps(resp.get("recommendation", {}), indent=2), file_name="recommendation.json")
# # #     else:
# # #         st.info("No evaluation reports available to compare.")





# # # # # Proposal_analysis_frontend_app.py
# # # # import streamlit as st
# # # # import requests
# # # # import json
# # # # import pandas as pd
# # # # import os


# # # # BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
# # # # st.set_page_config(page_title="RFP Proposal Analysis POC", layout="wide")
# # # # st.title("RFP Proposal Analysis — Streamlit POC")
# # # # st.info(f"Using backend URL: {BACKEND_URL}")

# # # # # -----------------------------
# # # # # Session state initialization
# # # # # -----------------------------
# # # # if 'rfp_uploaded' not in st.session_state:
# # # #     st.session_state['rfp_uploaded'] = None
# # # # if 'rules' not in st.session_state:
# # # #     st.session_state['rules'] = None
# # # # if 'scoring' not in st.session_state:
# # # #     st.session_state['scoring'] = None
# # # # if 'evaluations' not in st.session_state:
# # # #     st.session_state['evaluations'] = []  # list of {"filename":..., "result": {...}}
# # # # if 'page_idx' not in st.session_state:
# # # #     st.session_state['page_idx'] = 0

# # # # # -----------------------------
# # # # # Navigation
# # # # # -----------------------------
# # # # pages = ["Upload RFP", "Evaluate Proposals", "Compare Evaluations"]

# # # # def go_prev():
# # # #     st.session_state['page_idx'] = max(0, st.session_state['page_idx'] - 1)

# # # # def go_next():
# # # #     st.session_state['page_idx'] = min(len(pages)-1, st.session_state['page_idx'] + 1)

# # # # def end_session():
# # # #     # keep RFP/rules/scoring available on "front page" until user confirms clearing
# # # #     clear = st.confirm if hasattr(st, "confirm") else None
# # # #     # simply clear everything
# # # #     st.session_state.clear()
# # # #     st.success("Session cleared. Start again from RFP upload.")

# # # # st.sidebar.write(f"Current Page: {pages[st.session_state['page_idx']]}")
# # # # st.sidebar.button("Previous", on_click=go_prev)
# # # # st.sidebar.button("Next", on_click=go_next)
# # # # st.sidebar.button("End", on_click=end_session)

# # # # # -----------------------------
# # # # # Backend helper
# # # # # -----------------------------
# # # # def post_file(endpoint: str, files: dict, data: dict = None):
# # # #     url = f"{BACKEND_URL}{endpoint}"
# # # #     try:
# # # #         r = requests.post(url, files=files, data=data, timeout=300)
# # # #         r.raise_for_status()
# # # #         return r.json()
# # # #     except requests.exceptions.HTTPError as he:
# # # #         try:
# # # #             return r.json()
# # # #         except Exception:
# # # #             st.error(f"Backend request failed: {he}")
# # # #             return None
# # # #     except Exception as e:
# # # #         st.error(f"Backend request failed: {e}")
# # # #         return None

# # # # # -----------------------------
# # # # # Current page
# # # # # -----------------------------
# # # # page = pages[st.session_state['page_idx']]

# # # # # -----------------------------
# # # # # Page 1: Upload RFP
# # # # # -----------------------------
# # # # if page == "Upload RFP":
# # # #     st.header("1 — Upload RFP document (summary, rules, scoring)")
# # # #     uploaded = st.file_uploader("Upload RFP (pdf/docx/txt)", type=["pdf","docx","txt"])
# # # #     if uploaded:
# # # #         st.session_state['rfp_uploaded'] = uploaded.name
# # # #         st.info("Extracting summary, rules, scoring from RFP...")
# # # #         files = {"file": (uploaded.name, uploaded.getvalue())}
# # # #         resp = post_file("/extract_rfp", files)
# # # #         if not resp:
# # #             st.error("No response from backend.")
# # #         else:
# # #             if "error" in resp:
# # #                 st.error(resp["error"])
# # #             else:
# # #                 # store results in session
# # #                 st.session_state['rules'] = resp.get("rules", [])
# # #                 st.session_state['scoring'] = resp.get("scoring", [])
# # #                 # summary might be string
# # #                 st.subheader("Summary")
# # #                 st.write(resp.get("summary", "(no summary)"))

# # #                 st.subheader("Extracted Rules")
# # #                 st.json(st.session_state['rules'])
# # #                 st.subheader("Scoring")
# # #                 st.json(st.session_state['scoring'])

# # #                 if resp.get("errors"):
# # #                     st.warning("Partial extraction issues:")
# # #                     for e in resp.get("errors", []):
# # #                         st.write("-", e)

# # #                 st.download_button("Download rules JSON", json.dumps(st.session_state['rules'], indent=2),
# # #                                    file_name="rfp_rules.json")
# # #                 st.download_button("Download scoring JSON", json.dumps(st.session_state['scoring'], indent=2),
# # #                                    file_name="rfp_scoring.json")

# # # # -----------------------------
# # # # Page 2: Evaluate Proposals
# # # # -----------------------------
# # # elif page == "Evaluate Proposals":
# # #     st.header("2 — Upload proposals (max 3) and evaluate using RFP rules/scoring")
# # #     if not st.session_state.get('rules') or not st.session_state.get('scoring'):
# # #         st.warning("Please upload RFP on Page 1 to extract rules and scoring before evaluating proposals.")
# # #     uploaded_files = st.file_uploader("Upload proposals (max 3)", type=["pdf","docx","txt"], accept_multiple_files=True)
# # #     if uploaded_files:
# # #         if len(uploaded_files) > 3:
# # #             st.warning("Only first 3 files will be used")
# # #             uploaded_files = uploaded_files[:3]

# # #         evaluate = st.button("Evaluate proposals")
# # #         if evaluate:
# # #             all_results = []
# # #             rules_json = json.dumps(st.session_state.get('rules') or [])
# # #             scoring_json = json.dumps(st.session_state.get('scoring') or [])
# # #             with st.spinner("Evaluating proposals..."):
# # #                 for f in uploaded_files:
# # #                     files = {"file": (f.name, f.getvalue())}
# # #                     data = {"rules_json": rules_json, "scoring_json": scoring_json}
# # #                     resp = post_file("/evaluate_proposal", files, data=data)
# # #                     if resp is None:
# # #                         st.error(f"Backend did not return a response for {f.name}")
# # #                         continue
# # #                     if "error" in resp:
# # #                         st.error(f"Evaluation failed for {f.name}: {resp.get('error')}")
# # #                         all_results.append({"filename": f.name, "result": resp})
# # #                     else:
# # #                         all_results.append({"filename": f.name, "result": resp})
# # #             # save in session
# # #             st.session_state['evaluations'] = all_results
# # #             st.success("Evaluations complete")
# # #             for item in all_results:
# # #                 st.subheader(item['filename'])
# # #                 st.json(item['result'])
# # #                 csv_df = pd.json_normalize(item['result'].get('evaluations', []))
# # #                 if not csv_df.empty:
# # #                     st.download_button(f"Download {item['filename']} evaluation CSV", csv_df.to_csv(index=False),
# # #                                        file_name=f"{item['filename']}_evaluation.csv")
# # #                 # save evaluation JSON file locally option
# # #                 st.download_button(f"Download {item['filename']} evaluation JSON", json.dumps(item['result'], indent=2),
# # #                                    file_name=f"{item['filename']}_evaluation.json")

# # # # -----------------------------
# # # # Page 3: Compare Evaluations
# # # # -----------------------------
# # # elif page == "Compare Evaluations":
# # #     st.header("3 — Compare evaluation reports and produce final recommendation")
# # #     combined = st.session_state.get('evaluations', [])

# # #     uploaded_jsons = st.file_uploader("Or upload evaluation JSON files (from previous step)", accept_multiple_files=True, type=['json'])
# # #     if uploaded_jsons:
# # #         combined = []
# # #         for uj in uploaded_jsons:
# # #             try:
# # #                 combined.append({"filename": uj.name, "result": json.loads(uj.getvalue().decode())})
# # #             except Exception as e:
# # #                 st.error(f"Failed to parse {uj.name}: {e}")

# # #     if not combined:
# # #         st.info("No evaluation reports available. Run evaluations on Page 2 or upload previously generated evaluation JSONs.")
# # #     else:
# # #         st.write("Evaluations to compare:", [c['filename'] for c in combined])

# # #         # Build aggregate table
# # #         rows = {}
# # #         for c in combined:
# # #             for ev in c['result'].get('evaluations', []):
# # #                 req = ev.get('RFP Requirement/Criteria') or ev.get('RFP Requirement') or ev.get('Requirement') or 'Unknown'
# # #                 addressed = ev.get('Addressed') or ev.get('Addressed?') or ev.get('Satisfied') or ''
# # #                 score = ev.get('Score') or ev.get('Max Points') or ev.get('Points') or None
# # #                 if req not in rows:
# # #                     rows[req] = {}
# # #                 if score:
# # #                     rows[req][c['filename']] = f"{addressed} ({score})"
# # #                 else:
# # #                     rows[req][c['filename']] = f"{addressed}"

# # #         df = pd.DataFrame.from_dict(rows, orient='index').fillna('')
# # #         st.dataframe(df)
# # #         st.download_button("Download comparison CSV", df.to_csv(index=True), file_name="comparison.csv")
# # #         # Prepare files payload to send to backend's compare_reports endpoint
# # #         if st.button("Ask LLM for final recommendation & reasons"):
# # #             files = []
# # #             for c in combined:
# # #                 # send each evaluation as a file-like tuple (filename, bytes)
# # #                 files.append(("files", (c['filename'], json.dumps(c['result']).encode('utf-8'), "application/json")))

# # #             try:
# # #                 resp = requests.post(f"{BACKEND_URL}/compare_reports", files=files, timeout=300)
# # #                 resp.raise_for_status()
# # #                 comp = resp.json()
# # #             except Exception as e:
# # #                 st.error(f"Compare request failed: {e}")
# # #                 comp = None

# # #             if comp:
# # #                 if "error" in comp:
# # #                     st.error(comp["error"])
# # #                 else:
# # #                     st.subheader("LLM Recommendation / Analysis")
# # #                     st.json(comp.get("recommendation", {}))
# # #                     st.download_button("Download comparison CSV (from backend)", comp.get("csv", df.to_csv(index=True)), file_name="comparison_from_backend.csv")
# # #                     st.download_button("Download recommendation JSON", json.dumps(comp.get("recommendation", {}), indent=2), file_name="recommendation.json")

# # # # # Proposal_analysis_frontend_app_updated.py
# # # # import streamlit as st
# # # # import requests
# # # # import json
# # # # import pandas as pd
# # # # import os

# # # # BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
# # # # st.info(f"Using backend URL: {BACKEND_URL}")

# # # # st.set_page_config(page_title="RFP Proposal Analysis POC", layout="wide")
# # # # st.title("RFP Proposal Analysis — Streamlit POC")

# # # # # -----------------------------
# # # # # Session state initialization
# # # # # -----------------------------
# # # # if 'rfp' not in st.session_state:
# # # #     st.session_state['rfp'] = None
# # # # if 'rules' not in st.session_state:
# # # #     st.session_state['rules'] = None
# # # # if 'scoring' not in st.session_state:
# # # #     st.session_state['scoring'] = None
# # # # if 'evaluations' not in st.session_state:
# # # #     st.session_state['evaluations'] = []

# # # # if 'page_idx' not in st.session_state:
# # # #     st.session_state['page_idx'] = 0

# # # # # -----------------------------
# # # # # Navigation
# # # # # -----------------------------
# # # # pages = ["Upload RFP", "Evaluate Proposals", "Compare Evaluations"]

# # # # def go_prev():
# # # #     st.session_state['page_idx'] = max(0, st.session_state['page_idx'] - 1)

# # # # def go_next():
# # # #     st.session_state['page_idx'] = min(len(pages)-1, st.session_state['page_idx'] + 1)

# # # # def end_session():
# # # #     st.session_state.clear()
# # # #     st.success("Session cleared. Start again from RFP upload.")

# # # # st.sidebar.write(f"Current Page: {pages[st.session_state['page_idx']]}")
# # # # st.sidebar.button("Previous", on_click=go_prev)
# # # # st.sidebar.button("Next", on_click=go_next)
# # # # st.sidebar.button("End", on_click=end_session)

# # # # # -----------------------------
# # # # # Backend helper
# # # # # -----------------------------
# # # # def post_file(endpoint: str, files: dict, data: dict = None):
# # # #     url = f"{BACKEND_URL}{endpoint}"
# # # #     try:
# # # #         r = requests.post(url, files=files, data=data, timeout=300)
# # # #         r.raise_for_status()
# # # #         return r.json()
# # # #     except Exception as e:
# # # #         st.error(f"Backend request failed: {e}")
# # # #         return None

# # # # # -----------------------------
# # # # # Current page
# # # # # -----------------------------
# # # # page = pages[st.session_state['page_idx']]

# # # # # -----------------------------
# # # # # Page 1: Upload RFP
# # # # # -----------------------------
# # # # if page == "Upload RFP":
# # # #     st.header("1 — Upload RFP document")
# # # #     uploaded = st.file_uploader("Upload RFP (pdf/docx/txt)", type=["pdf","docx","txt"])
# # # #     if uploaded:
# # # #         st.session_state['rfp'] = uploaded
# # # #         st.info("Extracting summary, rules, and scoring from RFP...")
# # # #         files = {"file": (uploaded.name, uploaded.getvalue())}
# # # #         resp = post_file("/extract_rfp", files)
# # # #         if resp:
# # # #             st.session_state['rules'] = resp.get("rules", [])
# # # #             st.session_state['scoring'] = resp.get("scoring", [])
# # # #             st.subheader("Summary")
# # # #             st.write(resp.get("summary", "(no summary)"))

# # # #             st.subheader("Extracted Rules")
# # # #             st.json(st.session_state['rules'])
# # # #             st.subheader("Scoring")
# # # #             st.json(st.session_state['scoring'])

# # # #             st.download_button("Download rules JSON", json.dumps(st.session_state['rules'], indent=2),
# # # #                                file_name="rfp_rules.json")
# # # #             st.download_button("Download scoring JSON", json.dumps(st.session_state['scoring'], indent=2),
# # # #                                file_name="rfp_scoring.json")

# # # # # -----------------------------
# # # # # Page 2: Evaluate Proposals
# # # # # -----------------------------
# # # # elif page == "Evaluate Proposals":
# # # #     st.header("2 — Upload up to 3 proposals for evaluation")
# # # #     uploaded_files = st.file_uploader("Upload proposals (max 3)", type=["pdf","docx","txt"], accept_multiple_files=True)
# # # #     if uploaded_files:
# # # #         if len(uploaded_files) > 3:
# # # #             st.warning("Only first 3 files will be used")
# # # #             uploaded_files = uploaded_files[:3]

# # # #         evaluate = st.button("Evaluate proposals")
# # # #         if evaluate:
# # # #             all_results = []
# # # #             rules_json = json.dumps(st.session_state['rules']) if st.session_state['rules'] else None
# # # #             scoring_json = json.dumps(st.session_state['scoring']) if st.session_state['scoring'] else None
# # # #             if not rules_json or not scoring_json:
# # # #                 st.error("Rules and scoring must be available from RFP upload first.")
# # # #             else:
# # # #                 with st.spinner("Evaluating proposals..."):
# # # #                     for f in uploaded_files:
# # # #                         files = {"file": (f.name, f.getvalue())}
# # # #                         data = {"rules_json": rules_json, "scoring_json": scoring_json}
# # # #                         resp = post_file("/evaluate_proposal", files, data=data)
# # # #                         if resp:
# # # #                             all_results.append({"filename": f.name, "result": resp})
# # # #                 st.session_state['evaluations'] = all_results
# # # #                 st.success("Evaluation complete")
# # # #                 for item in all_results:
# # # #                     st.subheader(item['filename'])
# # # #                     st.json(item['result'])
# # # #                     csv_df = pd.json_normalize(item['result'].get('evaluations', []))
# # # #                     if not csv_df.empty:
# # # #                         st.download_button(f"Download {item['filename']} CSV", csv_df.to_csv(index=False),
# # # #                                            file_name=f"{item['filename']}_evaluation.csv")

# # # # # -----------------------------
# # # # # Page 3: Compare Evaluations
# # # # # -----------------------------
# # # # elif page == "Compare Evaluations":
# # # #     st.header("3 — Compare evaluation reports")
# # # #     combined = st.session_state.get('evaluations', [])

# # # #     uploaded_jsons = st.file_uploader("Or upload evaluation JSON files", accept_multiple_files=True, type=['json'])
# # # #     if uploaded_jsons:
# # # #         combined = []
# # # #         for uj in uploaded_jsons:
# # # #             combined.append({"filename": uj.name, "result": json.loads(uj.getvalue().decode())})

# # # #     if combined:
# # # #         st.write("Evaluations to compare:", [c['filename'] for c in combined])
# # # #         rows = {}
# # # #         for c in combined:
# # # #             for ev in c['result'].get('evaluations', []):
# # # #                 req = ev.get('RFP Requirement/Criteria', ev.get('RFP Requirement') or 'Unknown')
# # # #                 sat = ev.get('Addressed', 'Unknown')
# # # #                 reason = ev.get('Explanation/Evidence from Proposal', '')
# # # #                 if req not in rows:
# # # #                     rows[req] = {}
# # # #                 rows[req][c['filename']] = f"{sat}: {reason[:100]}"

# # # #         df = pd.DataFrame.from_dict(rows, orient='index')
# # # #         st.dataframe(df)
# # # #         st.download_button("Download comparison CSV", df.to_csv(index=True), file_name="comparison.csv")

# # # #         # Simple evaluation reasoning
# # # #         st.subheader("Final Evaluation / Recommendation")
# # # #         scores = {}
# # # #         for c in combined:
# # # #             count_yes = sum(1 for ev in c['result'].get('evaluations', [])
# # # #                             if ev.get('Addressed') in ['Yes','yes',True])
# # # #             scores[c['filename']] = count_yes
# # # #         if scores:
# # # #             best = max(scores, key=scores.get)
# # # #             st.write(f"**Best Proposal:** {best}")
# # # #             st.write(f"Reason: Satisfies highest number of RFP requirements ({scores[best]}).")
# # # #     else:
# # # #         st.info("No evaluation reports found. Upload JSON files or evaluate proposals first.")
