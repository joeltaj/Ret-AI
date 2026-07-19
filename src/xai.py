"""
Self-XAI module: RAG retrieval over the concept knowledge base + a local,
deterministic template narrator.

Faithfulness principle (from fgadr_concept_kb.yaml):
  The narrator must be conditioned on the CBM's ACTUAL concept activations.
  RAG supplies descriptive/clinical knowledge only; it must NOT decide which
  lesions are present, and it must NOT change the predicted grade.
"""

import yaml


# General, grade-mapped follow-up guidance. This is educational guidance based
# on typical DR management pathways, NOT a prescription for an individual patient.
GRADE_FOLLOWUP = {
    0: "No DR detected. Routine annual screening is generally recommended.",
    1: "Mild NPDR. Re-screen in ~12 months; monitor for progression.",
    2: "Moderate NPDR. Ophthalmology review is advisable; typical re-screen 6–12 months.",
    3: "Severe NPDR. Prompt referral to an ophthalmologist is advised; close monitoring "
       "(often every 2–4 months). Laser photocoagulation may be considered.",
    4: "Proliferative DR. Urgent ophthalmology referral; treatment such as pan-retinal "
       "photocoagulation and/or anti-VEGF is often indicated.",
}


def followup_for_grade(grade):
    """Return general, grade-mapped follow-up guidance (not a prescription)."""
    return GRADE_FOLLOWUP.get(grade, "Clinical correlation advised.")


def load_kb(kb_path):
    """Load the YAML KB and index concepts by their id (e.g. 'MA')."""
    with open(kb_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    concept_index = {}
    for entry in raw.get("concepts", []) or []:
        cid = entry.get("id")
        if cid:
            concept_index[cid] = entry

    return {
        "icdr_scale": raw.get("icdr_scale", {}),
        "concepts": concept_index,
        "references": raw.get("references", {}) or {},
    }


def _flatten_va(va):
    if isinstance(va, dict):
        return "; ".join(f"{k}: {v}" for k, v in va.items())
    return str(va)


def build_rag_context(result, kb):
    """Deterministic RAG: build the KB context for ONLY the fired concepts.

    Mirrors the reference notebook (Cell 27/28) so the LLM is conditioned solely
    on the CBM's actual output. Returns a dict of prompt-ready strings.
    """
    concepts = result["concepts"]
    active = result["active_concepts"]
    fired = [(c, concepts[c]) for c in active]

    blocks = []
    for cid, prob in fired:
        e = kb["concepts"].get(cid)
        if not e:
            continue
        blocks.append(
            f"CONCEPT {cid} ({e.get('name')}) - model confidence {prob:.2f}\n"
            f"  pathophysiology: {str(e.get('pathophysiology', '')).strip()}\n"
            f"  appearance: {_flatten_va(e.get('visual_appearance'))}\n"
            f"  grading_rule: {str(e.get('grading_rule', '')).strip()}\n"
            f"  appears at ICDR grades: {e.get('icdr_severity')}\n"
            f"  sources: {e.get('sources')}"
        )

    grade = result["grade"]
    grade_desc = kb["icdr_scale"].get(grade, kb["icdr_scale"].get(str(grade), ""))
    kb_context = (
        "\n\n".join(blocks) if blocks
        else "(No lesion concepts were fired by the model.)"
    )
    refs_lines = [
        f"{k}: {v.get('citation')}"
        for k, v in kb.get("references", {}).items()
        if isinstance(v, dict) and v.get("citation")
    ]
    fired_str = ", ".join(f"{c} ({p:.2f})" for c, p in fired) or "none"

    return {
        "fired": fired,
        "kb_context": kb_context,
        "refs_context": "\n".join(refs_lines),
        "refs_lines": refs_lines,
        "grade_desc": grade_desc,
        "fired_str": fired_str,
    }


def template_narrative_text(result, kb):
    """Deterministic KB narrative (fallback when the LLM is unavailable).

    Same 3-section structure the LLM is asked to produce, built directly from
    the fired concepts and the knowledge base.
    """
    ctx = build_rag_context(result, kb)
    fired = ctx["fired"]
    grade = result["grade"]
    grade_name = result["grade_name"]

    out = ["1. Findings:"]
    if fired:
        for cid, prob in fired:
            e = kb["concepts"].get(cid, {})
            out.append(
                f"   - {cid} ({e.get('name')}, confidence {prob:.2f}): "
                f"{str(e.get('pathophysiology', '')).strip()} "
                f"Appearance -> {_flatten_va(e.get('visual_appearance'))}."
            )
    else:
        out.append("   - No FGADR lesion concepts were detected by the model.")

    out += ["", f"2. Concept -> grade reasoning: predicted ICDR grade {grade} "
                f"({grade_name}) - {ctx['grade_desc']}"]
    if fired:
        for cid, _ in fired:
            gr = str(kb["concepts"].get(cid, {}).get("grading_rule", "")).strip()
            if gr:
                out.append(f"   - {cid}: {gr}")
    else:
        out.append("   - Absence of the six FGADR lesions is consistent with the grade.")

    out += ["", "3. Sources:"]
    src = []
    for cid, _ in fired:
        src.extend(kb["concepts"].get(cid, {}).get("sources", []) or [])
    for s in sorted(set(src)):
        out.append(f"   - {s}")
    for rl in ctx["refs_lines"]:
        out.append(f"   - {rl}")
    return "\n".join(out)


def retrieve(kb, active_concepts):
    """Return the KB chunks for the concepts the CBM actually fired."""
    return {
        cid: kb["concepts"][cid]
        for cid in active_concepts
        if cid in kb["concepts"]
    }


def _first_sentence(text):
    if not text:
        return ""
    text = " ".join(str(text).split())
    for sep in (". ", ".\n"):
        if sep in text:
            return text.split(sep)[0].strip() + "."
    return text.strip()


def narrate(result, kb):
    """Produce a structured, faithful explanation from CBM output + KB.

    `result` is the dict returned by inference.predict().
    Returns a dict with summary / evidence / clinical_interpretation /
    limitations / caveats.
    """
    active = result["active_concepts"]
    concepts = result["concepts"]
    knowledge = retrieve(kb, active)

    # ---- summary -------------------------------------------------------
    conf = result["grade_probability"]
    icdr_desc = kb["icdr_scale"].get(result["grade"], "")
    summary = (
        f"The concept-bottleneck model predicts "
        f"{result['grade_name']} (grade {result['grade']}) "
        f"with {conf * 100:.1f}% confidence."
    )
    if icdr_desc:
        summary += f" ICDR reference: {icdr_desc}"

    # ---- evidence: which concepts fired, with probabilities ------------
    if active:
        parts = []
        for cid in active:
            entry = knowledge.get(cid, {})
            name = entry.get("name", cid)
            parts.append(f"{name} ({cid}): {concepts[cid] * 100:.0f}%")
        evidence = (
            "The grade is driven by these detected retinal concepts — "
            + "; ".join(parts)
            + "."
        )
    else:
        evidence = (
            "No retinal lesion concept exceeded the detection threshold "
            f"({result['threshold']:.2f}); the prediction reflects the "
            "absence of detectable lesions."
        )

    # ---- clinical interpretation: concept -> grade chain ---------------
    interp_parts = []
    for cid in active:
        entry = knowledge.get(cid, {})
        rule = _first_sentence(entry.get("grading_rule"))
        patho = _first_sentence(entry.get("pathophysiology"))
        name = entry.get("name", cid)
        line = f"{name}"
        if patho:
            line += f" — {patho}"
        if rule:
            line += f" Grading implication: {rule}"
        interp_parts.append(line)
    clinical_interpretation = (
        " ".join(interp_parts)
        if interp_parts
        else "No lesion-based reasoning applies; findings are within normal limits."
    )

    # ---- limitations & caveats ----------------------------------------
    limitations = (
        "This inference uses image-derived concepts only (strict CBM). "
        "Quantitative lesion measurements and segmentation masks were not used. "
        "Concept probabilities near the threshold should be treated as uncertain."
    )
    caveats = (
        "This is a decision-support explanation, not a diagnosis. "
        "Clinical confirmation requires ophthalmological assessment, and the "
        "predicted grade must not be overridden by this narrative."
    )

    return {
        "summary": summary,
        "evidence": evidence,
        "clinical_interpretation": clinical_interpretation,
        "limitations": limitations,
        "caveats": caveats,
        "retrieved_knowledge": knowledge,
    }
