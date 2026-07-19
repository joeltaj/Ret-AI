# Reti-AI Dashboard — UI/UX Redesign Recommendations

> Generated from UX critique of current `Reti-AI: Explainable Eye Disease Diagnosis` dashboard.
> Overall score: **5.8 / 10**. Priority order: Critical → Notable → Enhancements.

---

## 🔴 Critical Issues

### 1. Broken Reading Order / Visual Hierarchy

**Problem:**
User's eye has no clear path. Currently: top-left fundus images → top-right narrative → bottom-left quantitative → bottom-right classification. The most clinically important output (Final Classification) is hidden in the bottom-right corner.

**Fix:**
Restructure into a 3-zone layout:

```
┌─────────────────────────────────────────────────────┐
│  [Patient Banner — sticky, full-width]               │
├──────────────┬───────────────────┬──────────────────┤
│  INPUT ZONE  │  ANALYSIS ZONE   │  VERDICT ZONE    │
│  Fundus imgs │  CBM concepts    │  Diagnosis        │
│  Controls    │  Grade chart     │  Confidence       │
│              │  Quantitative    │  Severity badge   │
│              │                  │  Recommended Rx   │
├──────────────┴───────────────────┴──────────────────┤
│  [Narrative Explanation — collapsible, full-width]   │
└─────────────────────────────────────────────────────┘
```

**Verdict zone (right column) must be the largest, most prominent visual element on the page.**

---

### 2. Semantically Incorrect Color Usage (Red)

**Problem:**
"Proliferative Diabetic Retinopathy", "61%", and "PROLIFERATIVE" are all rendered in bright red as plain text labels. In clinical UI, red = urgent alert/action required. Using it as a category label causes alarm fatigue and potential misinterpretation.

**Fix:**
- Use semantic color consistently: red only for actionable alerts (e.g., "Urgent referral required" button/banner)
- Replace raw red text with **tinted severity badges**:

```
Severity: [ PROLIFERATIVE ]   ← background: #FEE2E2, text: #991B1B, border: #FCA5A5
```

- One red element max = the "Urgent Ophthalmology Referral" call-to-action, if clinically indicated
- Diagnosis label: use neutral dark text + colored badge, not colored text on dark background

---

### 3. Narrative Explanation — Wall of Text

**Problem:**
Right-side narrative panel contains long running prose paragraphs. Clinicians need scannable, structured information, not an essay. Key findings are buried mid-paragraph.

**Fix — restructure narrative into 3 tiers:**

```
KEY DRIVERS
  [HE 90%]  [EX 71%]  [MA 29%]   ← concept chips, color-coded

CLINICAL IMPLICATION
  Intraretinal hemorrhage indicates at least Moderate NPDR.
  Hard exudate contributes to macular risk.

RECOMMENDED ACTION                              ← bold callout box
  Urgent ophthalmology referral. Consider
  pan-retinal photocoagulation and/or anti-VEGF.

  [ Read full explanation ▾ ]                  ← collapsible prose
```

---

## 🟡 Notable Issues

### 4. Grade Distribution Chart — Ambiguous & Inaccessible

**Problem:**
- X-axis labels are missing or cut off
- Bars for Grade 2 and 3 use near-identical colors — fails for color-blind users (~8% of males)
- Ordinal data (DR grades 0–4) should not use categorical coloring

**Fix:**
- Add explicit bar labels: Grade 0, Grade 1, Grade 2, Grade 3, Grade 4
- Use a **sequential color ramp** (yellow → orange → red) not categorical colors
- Include percentage labels directly on/above each bar
- Add a small colorblind-safe pattern fill as secondary encoding (e.g., hatching for severe grades)

```
Grade:   0      1      2      3      4
Color:  #FEF9C3 #FDE68A #FDBA74 #F97316 #DC2626
```

---

### 5. Cognitive Overload — Too Many Competing Panels

**Problem:**
8+ active panels simultaneously: 3 fundus images, overlay legend, lesion size bar, quantitative metrics, narrative, grade chart, final classification, patient header, threshold input, model status. This exceeds the recommended 5±2 cognitive chunks.

**Fix — implement progressive disclosure:**

**Option A — Tab layout:**
```
[ Overview ]  [ Analysis ]  [ Explanation ]  [ Report ]
```

**Option B — Default summary view with detail toggle:**
```
Summary (default)          Detailed view (toggle)
─────────────────          ──────────────────────
Diagnosis verdict          + CBM concept scores
Top 2 concepts             + Lesion size distribution
Recommended action         + Quantitative analysis table
                           + Full narrative text
```

---

### 6. Patient Identity Banner — Too Small / Not Prominent

**Problem:**
Patient data (ID, Name, Age, Gender, DOB) is in small text in the top-right corner, visually competing with the model status indicator. Patient identity confirmation is the most safety-critical step in clinical workflows.

**Fix:**
- Sticky full-width patient banner at the very top, always visible
- Large name, clear ID, age/gender in secondary text
- Color-coded session status: green = active session, gray = archived
- Optional: alert if patient ID does not match loaded image metadata

```
┌─────────────────────────────────────────────────────────────────┐
│  👤  Rudi Santoso  ·  RD-240901  ·  58 y/o Male  ·  15/03/1966 │
│                                              ● Active session    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🟢 What's Already Working — Keep These

| Element | Why it works |
|---|---|
| Dark theme as default | Reduces eye strain in dim clinical environments |
| 3-view fundus display (original / detection / mask) | Clinically valuable, no redundancy |
| CBM concept scores (HE 90%, EX 71%, etc.) | Actionable XAI — directly maps to ICDR grading criteria |
| Overlay legend (MA, HE, EX, SE, IRMA, NV) | Clean, color-coded, doesn't clutter the image |
| Model status indicator (CUDA · seg on) | Useful for technical operators |
| Threshold control | Good for research/calibration use cases |

---

## Implementation Priority

| # | Issue | Effort | Impact |
|---|---|---|---|
| 1 | Restructure 3-zone layout + verdict prominence | High | Critical |
| 2 | Semantic color fix (red → tinted badges) | Low | Critical |
| 3 | Narrative → structured 3-tier format | Medium | High |
| 4 | Patient banner — sticky + prominent | Low | High |
| 5 | Progressive disclosure (tabs or toggle) | Medium | High |
| 6 | Grade chart — sequential color + labels | Low | Medium |

---

## Component References (for Claude Code)

If rebuilding with a component library, these map to common patterns:

- **Verdict zone** → `Card` with `Alert` variant for severity, `Badge` for confidence
- **Concept chips** → `Tag` / `Badge` with color props
- **Narrative tiers** → `Accordion` or `Disclosure` component
- **Patient banner** → Sticky `Header` / `Toolbar`
- **Grade chart** → `BarChart` with ordinal color scale (e.g., Recharts `Cell` with custom fill array)
- **Progressive disclosure** → `Tabs` component or `Switch`/`Toggle` for detail view

---

*Critique basis: UX review of Reti-AI dashboard screenshot, July 2026.*
*Focus: clinical usability, information hierarchy, semantic color, cognitive load.*
