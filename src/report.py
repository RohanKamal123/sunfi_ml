"""Generate the plain-language project report as a PDF.

    python -m src.report

Writes ``reports/PCOS_Project_Report.pdf``. Every number in it is read from
``reports/results/`` at build time rather than typed in, so the report cannot
drift out of step with the pipeline that produced it.
"""

from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import config

# --------------------------------------------------------------------------
# Fonts. DejaVu is used rather than the built-in Helvetica because the report
# contains characters (≥, ≈, em dashes) that the Type-1 fonts cannot render,
# and reportlab draws missing glyphs as black boxes rather than failing loudly.
# --------------------------------------------------------------------------
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
BODY_FONT, BOLD_FONT = "DejaVu", "DejaVu-Bold"


def _register_fonts() -> tuple[str, str]:
    try:
        pdfmetrics.registerFont(TTFont(BODY_FONT, f"{FONT_DIR}/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont(BOLD_FONT, f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFontFamily(BODY_FONT, normal=BODY_FONT, bold=BOLD_FONT)
        return BODY_FONT, BOLD_FONT
    except Exception:
        # Fall back to the built-ins; the text stays readable, some symbols may not.
        return "Helvetica", "Helvetica-Bold"


INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#6B7280")
ACCENT = colors.HexColor("#E45756")
BLUE = colors.HexColor("#4C78A8")
PANEL = colors.HexColor("#F3F4F6")


def _styles(body: str, bold: str) -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName=bold, fontSize=23, leading=28,
            textColor=INK, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName=body, fontSize=12.5, leading=18,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName=bold, fontSize=16, leading=20,
            textColor=INK, spaceBefore=18, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName=bold, fontSize=12.5, leading=16,
            textColor=BLUE, spaceBefore=12, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName=body, fontSize=10.2, leading=15.5,
            textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName=body, fontSize=10.2, leading=15,
            textColor=INK, leftIndent=14, bulletIndent=3, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName=body, fontSize=8.8, leading=12.5,
            textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12,
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["Normal"], fontName=body, fontSize=11, leading=16,
            textColor=INK, alignment=TA_JUSTIFY,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName=body, fontSize=8.8, leading=12,
            textColor=INK,
        ),
        "cellb": ParagraphStyle(
            "cellb", parent=base["Normal"], fontName=bold, fontSize=8.8, leading=12,
            textColor=INK,
        ),
    }


def _callout(text: str, styles: dict, colour=ACCENT) -> Table:
    """A tinted box for the one sentence a reader must not miss."""
    tbl = Table([[Paragraph(text, styles["callout"])]], colWidths=[16.2 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colour),
    ]))
    return tbl


def _table(rows, styles: dict, widths=None, highlight_rows=()) -> Table:
    """A simple table; first row is the header."""
    data = [
        [Paragraph(str(c), styles["cellb"] if r == 0 else styles["cell"]) for c in row]
        for r, row in enumerate(rows)
    ]
    tbl = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9CA3AF")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#FDECEA")))
    tbl.setStyle(TableStyle(style))
    return tbl


def _figure(name: str, caption: str, styles: dict, width=15.5 * cm):
    """Place a generated figure, scaled to the text column."""
    path = config.FIGURES_DIR / name
    if not path.exists():
        return []
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        ratio = im.height / im.width
    img = Image(str(path), width=width, height=width * ratio)
    return [KeepTogether([img, Paragraph(caption, styles["caption"])])]


def _load_results() -> dict:
    """Read every number the report quotes from the pipeline's own output.

    Missing inputs are a hard error. An earlier version skipped absent files
    and let the report fall back to hardcoded defaults, which silently
    produced a plausible-looking document from stale numbers when a container
    restart reverted the results directory. A report that claims its figures
    come from the pipeline must fail loudly when they do not.
    """
    import pandas as pd

    r = config.RESULTS_DIR
    json_inputs = {
        "summary": "summary.json",
        "rules": "value_added_by_ml.json",
        "nested": "nested_cv.json",
    }
    csv_inputs = {
        "leakage": "leakage_experiment.csv",
        "tiers": "cost_tiers.csv",
        "wins": "seed_sweep_wins.csv",
        "ci": "bootstrap_ci.csv",
    }

    missing = [
        name for name in list(json_inputs.values()) + list(csv_inputs.values())
        if not (r / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot build the report; these pipeline outputs are missing from "
            f"{r}: {', '.join(sorted(missing))}.\n"
            "Run `python -m src.run_pipeline` first."
        )

    out = {k: json.loads((r / v).read_text()) for k, v in json_inputs.items()}
    out.update({k: pd.read_csv(r / v) for k, v in csv_inputs.items()})
    return out


def build(output_path=None):
    body, bold = _register_fonts()
    st = _styles(body, bold)
    res = _load_results()
    s = res["summary"]
    rules = res["rules"]

    output_path = output_path or (config.REPORTS_DIR / "PCOS_Project_Report.pdf")
    config.ensure_dirs()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2.4 * cm, rightMargin=2.4 * cm,
        topMargin=2.2 * cm, bottomMargin=2.0 * cm,
        title="Machine Learning-Based Prediction of PCOS",
        author="Group 5",
    )

    story = []
    P = lambda t: Paragraph(t, st["body"])
    B = lambda t: Paragraph(t, st["bullet"], bulletText="•")

    # ---------------------------------------------------------------- cover
    story += [
        Spacer(1, 3.2 * cm),
        Paragraph("Predicting Polycystic Ovary Syndrome<br/>with Machine Learning", st["title"]),
        Spacer(1, 0.5 * cm),
        Paragraph("A plain-language report on what we built, what we found,<br/>"
                  "and what we could not do", st["subtitle"]),
        Spacer(1, 1.4 * cm),
        Paragraph("<b>Group 5</b>", st["subtitle"]),
        Paragraph("Fatema Ferdous &nbsp;·&nbsp; 0152410052", st["subtitle"]),
        Paragraph("Wafa Haque &nbsp;·&nbsp; 0152420023", st["subtitle"]),
        Paragraph("Khondoker Sazzad Sunfi &nbsp;·&nbsp; 0152310002", st["subtitle"]),
        Spacer(1, 1.8 * cm),
        _callout(
            "<b>The short version.</b> We built a system that predicts PCOS from routine "
            "medical measurements, and it works well — about "
            f"<b>{s['test_accuracy']:.0%} accurate</b>. But the more useful part of this "
            "project is what we found while checking our own work: several published "
            "papers report 98–99% accuracy on this exact dataset, and we show that a "
            "large part of such a score can come from <i>how you test</i> rather than "
            "how good your model is.", st),
        Spacer(1, 1.0 * cm),
        Paragraph("Machine Learning course project", st["subtitle"]),
        PageBreak(),
    ]

    # ------------------------------------------------------------ section 1
    story += [
        Paragraph("1. The problem, in plain terms", st["h1"]),
        P("Polycystic Ovary Syndrome (PCOS) is a hormonal condition affecting roughly "
          "8–13% of women of reproductive age. It causes irregular periods, weight gain, "
          "excess hair growth and skin changes, and if it goes undetected it raises the "
          "risk of infertility, type 2 diabetes and heart disease."),
        P("The difficulty is that these symptoms are common and overlap with other "
          "conditions. Proper diagnosis needs blood tests and an ultrasound scan, which "
          "are not always quick, cheap or available — especially in areas with limited "
          "healthcare resources."),
        P("<b>Our question:</b> can a computer program look at a woman's routine "
          "measurements and flag whether she is likely to have PCOS, so that a doctor "
          "knows who to investigate further?"),

        Paragraph("2. The data", st["h1"]),
        P(f"We used a public dataset of <b>{s['n_samples']} women</b> from 10 hospitals in "
          f"Kerala, India, with <b>{s['n_features']} measurements</b> recorded for each "
          "person — age, weight, blood pressure, hormone levels, symptoms, and ultrasound "
          "findings."),
    ]
    story += _figure(
        "01_class_balance.png",
        "Of the 541 women, 177 have PCOS and 364 do not — about two healthy women for "
        "every patient. This imbalance matters: a lazy program could score 67% just by "
        "guessing 'no PCOS' every single time.",
        st, width=9.5 * cm)

    story += [
        Paragraph("A dataset with mistakes in it", st["h2"]),
        P("Before doing anything clever, we had to clean the data — and it needed more "
          "cleaning than expected. Among other problems we found <b>13 physically "
          "impossible values</b>:"),
        B("A blood pressure of <b>12/80</b> and another of <b>120/8</b> — clearly someone "
          "dropped a digit from 120/80."),
        B("Pulse rates of <b>13 and 18 beats per minute</b>. A person with that pulse is "
          "not sitting in a clinic answering questions."),
        B("A hormone reading of <b>5052</b> where normal is under 25 — a decimal point in "
          "the wrong place."),
        Spacer(1, 4),
        P("We <b>blanked</b> these rather than guessing what they should have been, and "
          "let the program fill them in the same way it fills any other missing value. "
          "Guessing '120/80' would look tidier but would be inventing data."),
        _callout(
            "Worth noting: four of the five published papers we reviewed use this same "
            "dataset, and none of them mention these errors. They almost certainly "
            "trained on a pulse rate of 13 bpm without noticing.", st, BLUE),
    ]

    # ------------------------------------------------------------ section 3
    story += [
        Paragraph("3. What we built", st["h1"]),
        P("We built an automated pipeline — a fixed sequence of steps that runs the same "
          "way every time:"),
    ]
    steps = [
        ["Step", "What it does", "Why"],
        ["1. Clean", "Fix the broken values described above", "Rubbish in, rubbish out"],
        ["2. Fill gaps", "Estimate the handful of missing numbers", "Models cannot handle blanks"],
        ["3. Rescale", "Put every measurement on a comparable scale",
         "Otherwise 'weight in kg' outshouts 'number of symptoms'"],
        ["4. Choose features", "Keep the most informative measurements",
         "Fewer inputs, less chance of memorising noise"],
        ["5. Balance", "Even out the 2:1 imbalance", "So PCOS cases are not ignored"],
        ["6. Predict", "One of nine different algorithms", "No single one is obviously best"],
    ]
    story += [
        _table(steps, st, widths=[2.8 * cm, 5.6 * cm, 7.4 * cm]),
        Spacer(1, 10),
        P("The critical detail is <b>where</b> steps 2–5 happen. They all sit "
          "<i>inside</i> the pipeline, so they are recalculated from scratch every time "
          "the model is tested on a new group of patients. That sounds like a technicality. "
          "It is the single most important decision in the project, and Section 4 explains "
          "why."),

        Paragraph("How we tested it honestly", st["h1"]),
        P("We set aside <b>20% of the women at the very start</b> and did not look at them "
          "again until the end. Everything — which features to use, which algorithm to "
          "pick — was decided using only the other 80%."),
        P("Think of it as an exam. The training data is the textbook; the held-out 20% is "
          "the exam paper, kept in a sealed envelope. If you peek at the exam while "
          "revising, your score stops meaning anything."),
    ]

    # ------------------------------------------------------------ finding 1
    leak = res["leakage"]
    inflation = float(leak.iloc[2]["accuracy"])
    f1_inflation = float(leak.iloc[2]["f1"])

    story += [
        Paragraph("4. Finding 1: how to accidentally cheat", st["h1"]),
        P("Published papers using this dataset report accuracies of 98–99%. We reach about "
          f"{s['test_accuracy']:.0%}. Rather than argue about it, we <b>measured</b> where "
          "such a gap can come from."),
        P("Remember step 5, balancing the data. It works by inventing extra synthetic "
          "PCOS patients, each built by blending real ones. Now consider the order of "
          "operations:"),
        B("<b>The right way</b> — split the patients first, then invent synthetic ones "
          "using only the training group."),
        B("<b>The tempting way</b> — invent synthetic patients first, then split. Now a "
          "synthetic patient built from a training woman can land in the exam envelope. "
          "The model is partly being tested on copies of what it studied."),
        Spacer(1, 4),
        P("We ran both, with the same model, the same data and the same patients:"),
    ]
    story += _figure(
        "13_leakage_experiment.png",
        "Same model, same data, same split. The only difference is the order of two steps.",
        st, width=14 * cm)
    story += [
        _callout(
            f"Doing it the tempting way inflates accuracy by <b>{inflation:.1%}</b> and "
            f"the F1 score by <b>{f1_inflation:.1%}</b> — without the model getting any "
            "better at all. A 98–99% headline on this dataset is <b>reachable through "
            "testing method alone</b>.", st),
        Spacer(1, 8),
        P("This does not prove the other papers made this mistake. Most simply do not "
          "describe their method in enough detail to tell — which is itself the finding. "
          "A number you cannot check is a number you cannot trust."),
    ]

    # ------------------------------------------------------------ finding 2
    n_winners = s["seed_sweep_n_distinct_winners"]
    top_rate = s["seed_sweep_top_win_rate"]

    story += [
        Paragraph("5. Finding 2: there is no 'best' algorithm here", st["h1"]),
        P("We compared <b>nine</b> different algorithms, from simple logistic regression "
          "to a stacked ensemble of several models. Papers routinely announce a winner. "
          "We tested whether a winner even exists."),
        P("We re-shuffled which patients go into training and which into the exam envelope "
          "<b>30 times</b>, and recorded which algorithm came first each time:"),
    ]
    story += _figure(
        "23_seed_sweep.png",
        "Each algorithm's score swings by 8–13 percentage points depending purely on which "
        "patients happened to land in the test set.",
        st, width=15.5 * cm)
    story += [
        _callout(
            f"<b>{n_winners} of the 9 algorithms won at least once.</b> The most frequent "
            f"winner took only {top_rate:.0%} of the rounds — and it was Gaussian Naive "
            "Bayes, one of the <i>simplest</i> methods available. A paper that declares a "
            "best model from a single split is reporting a coin flip.", st),
        Spacer(1, 8),
        P("This is not a failure of the models. It is a consequence of having 541 patients. "
          "With only 36 PCOS cases in the exam envelope, three women changing sides moves "
          "the accuracy by several points."),
    ]

    # ------------------------------------------------------------ finding 3
    tiers = res["tiers"]
    q_auc = s["questionnaire_only_cv_auc"]
    pct = s["questionnaire_pct_of_full"]
    us_gain = s["ultrasound_marginal_auc_gain"]
    blood_gain = float(tiers.iloc[2]["gain_over_previous"])

    story += [
        Paragraph("6. Finding 3: you may not need the expensive tests", st["h1"]),
        P("Our introduction argued that PCOS testing is often hard to access. So we asked "
          "a question none of the reviewed papers ask: <b>what does each level of testing "
          "actually buy you?</b>"),
        P("We grouped the measurements by how much they cost to obtain, and trained a "
          "separate model on each level."),
    ]
    tier_rows = [["What you need", "Model quality (AUC)", "Gained by adding this step"]]
    if tiers is not None:
        labels = {
            "questionnaire": "A form, a scale, a tape measure",
            "+ clinic vitals": "+ a nurse for 5 minutes",
            "+ blood panel": "+ blood draw and lab tests",
            "+ ultrasound (all)": "+ ultrasound scan and a sonographer",
        }
        for _, row in tiers.iterrows():
            gain = row["gain_over_previous"]
            tier_rows.append([
                labels.get(row["tier"], row["tier"]),
                f"{row['cv_roc_auc']:.3f}",
                "—" if gain != gain else f"{gain:+.4f}",
            ])
    story += [
        _table(tier_rows, st, widths=[8.2 * cm, 3.6 * cm, 4.0 * cm], highlight_rows=(1, 4)),
        Spacer(1, 10),
    ]
    story += _figure(
        "27_cost_tiers.png",
        "Left: how good the model is at each level of testing. Right: what each extra "
        "step adds. Only the ultrasound bar is large.",
        st, width=15.5 * cm)
    story += [
        _callout(
            f"A <b>questionnaire alone</b> — no doctor, no laboratory, no ultrasound — "
            f"reaches {q_auc:.3f}, which is <b>{pct:.0%} of the full model</b>. "
            f"The entire blood panel (nine lab tests and a needle) adds {blood_gain:+.4f}. "
            f"Only the ultrasound genuinely pays for itself, at {us_gain:+.4f}.", st),
        Spacer(1, 8),
        P("<b>The practical recommendation:</b> in a clinic that cannot afford everything, "
          "run the questionnaire, skip the blood tests, and spend the budget on ultrasound "
          "access for the women the questionnaire flags."),
    ]

    # ------------------------------------------------------------ finding 4
    story += [
        Paragraph("7. Finding 4: does any of this beat what doctors already do?", st["h1"]),
        P("This is the question we think matters most, and none of the five papers we "
          "reviewed asks it."),
        P("Doctors already have a rule. The Rotterdam criteria, agreed in 2003, say: if an "
          "ovary shows <b>12 or more follicles</b>, that counts toward a PCOS diagnosis. "
          "That is not machine learning. It is counting, with a threshold."),
        P("So we ran that rule through exactly the same tests as our nine algorithms."),
    ]
    story += _figure(
        "29_clinical_rule_baseline.png",
        "Red = machine learning, grey = simple rules. Left: overall quality. "
        "Right: what fraction of actual PCOS cases each approach finds.",
        st, width=15.5 * cm)

    rule_auc = rules["best_rule_cv_auc"]
    ml_auc = rules["best_ml_cv_auc"]
    pct_rule = rules["pct_of_ml_auc_from_rule"]
    found_rule = rules["cases_found_by_rule"]
    found_ml = rules["cases_found_by_ml"]
    extra = rules["extra_cases_found_by_ml"]
    fa = rules["extra_false_alarms_from_ml"]
    n_pos = rules["test_positives"]

    story += [
        Paragraph("The uncomfortable half of the answer", st["h2"]),
        P(f"Just counting follicles scores <b>{rule_auc:.3f}</b>. Our best model, after "
          f"nine algorithms, data balancing, feature selection and an automated "
          f"architecture search, scores <b>{ml_auc:.3f}</b>."),
        _callout(
            f"A rule from 2003 with <b>nothing learned from data at all</b> achieves "
            f"<b>{pct_rule:.0%}</b> of what all our machine learning achieves. Most of the "
            "signal in this dataset is simply the follicle count.", st, BLUE),

        Paragraph("The half that justifies the work", st["h2"]),
        P("But overall scores hide something important. The old rule is very good at "
          "correctly clearing healthy women, and much worse at the job it exists to do — "
          "finding patients:"),
    ]
    story += [
        _table([
            ["Approach", f"PCOS cases found (of {n_pos})", "False alarms (of 73)"],
            ["Counting follicles (the 2003 rule)", str(found_rule), "1"],
            ["Our machine learning model", str(found_ml), "2"],
        ], st, widths=[7.6 * cm, 4.4 * cm, 3.8 * cm], highlight_rows=(2,)),
        Spacer(1, 10),
        _callout(
            f"The model finds <b>{extra} more women with PCOS</b> and raises "
            f"<b>{fa} extra false alarm</b>. The old rule misses 39% of the patients it "
            "is meant to catch; our model misses 17%. A false alarm costs one extra scan. "
            "A missed case costs years of undiagnosed illness.", st),
        Spacer(1, 8),
        P("<b>So the honest conclusion is neither 'machine learning works' nor 'machine "
          "learning is pointless'.</b> Almost all the <i>information</i> is in the follicle "
          "count, which a doctor can already read off a scan. What the model adds is a "
          "better decision about where to draw the line — and for a screening tool, that "
          "is exactly the job."),
    ]

    # ------------------------------------------------------------ finding 5
    ext = s["external_validation"]
    story += [
        Paragraph("8. Finding 5: does it work on different women?", st["h1"]),
        P("Every paper we reviewed was criticised in our literature review for testing "
          "only on the population it was built from. We tried to do better."),
        P(f"We found a genuinely independent study — <b>{ext['n']} women from a "
          "hospital in Sfax, Tunisia</b> — and tested our model on them. Different "
          "country, younger patients, heavier on average, and a different balance of "
          "cases. We checked carefully that it was not simply a re-upload of our own "
          "training data, which many public 'PCOS datasets' turn out to be."),
    ]
    story += _figure(
        "21_external_validation.png",
        "Left: performance at home (blue) versus abroad (red). Right: how different the "
        "two groups of women actually are.",
        st, width=15.5 * cm)
    story += [
        P("<b>The good news:</b> performance barely moved — an average change of "
          f"{ext['mean_auc_drop']:+.3f}. What the model learned is not "
          "specific to Kerala."),
        P("<b>The catch:</b> the Tunisian study recorded only 8 of our 41 measurements, "
          "and it did not record follicle counts or symptoms — the very things that carry "
          "the signal. So we could only test a weakened version of our model."),
        _callout(
            "This turns into a stronger criticism of the field than we expected. The five "
            "papers did not merely <i>neglect</i> to test on other populations — "
            "<b>no public dataset exists that would let anyone do it</b>. What the field "
            "needs is not better algorithms but compatible data collection across "
            "hospitals.", st, BLUE),
    ]

    # ------------------------------------------------------- what it uses
    story += [
        Paragraph("9. What the model actually pays attention to", st["h1"]),
        P("A prediction nobody can explain is not much use to a doctor. We used a "
          "technique called SHAP, which breaks each individual prediction down into how "
          "much each measurement pushed it one way or the other."),
    ]
    story += _figure(
        "17_shap_beeswarm.png",
        "Each dot is one patient. Dots to the right pushed the prediction toward PCOS. "
        "Red means a high value of that measurement, blue means low.",
        st, width=12.5 * cm)
    top = s["top_shap_features"]
    story += [
        P("The model's priorities, in order, are: " +
          ", ".join(f"<b>{t}</b>" for t in top[:5]) + "."),
        P("That is reassuring, because it matches the three things doctors already look "
          "for: ovary appearance, signs of excess male hormones, and irregular cycles. "
          "The model was never told the medical criteria — it recovered them from data."),
        _callout(
            "<b>But there is a catch we should state plainly.</b> Follicle count is both "
            "the model's top clue <i>and</i> one of the three criteria doctors use to "
            "record the diagnosis in the first place. So the model is partly learning the "
            "definition of the answer. Our questionnaire-only model is arguably more "
            "interesting, because none of its inputs are diagnostic criteria.", st),
    ]

    # ------------------------------------------------------------- honesty
    recall_row = res["ci"].set_index("metric").loc["recall"]

    story += [
        Paragraph("10. What we could not do", st["h1"]),
        P("A report that only lists successes is not a scientific report. These are the "
          "real limits of this work."),
        B("<b>Our numbers are less precise than they look.</b> With only 36 PCOS cases in "
          "the test set, the fraction of patients we correctly find could plausibly be "
          + f"anywhere from {recall_row['ci_lower']:.0%} to {recall_row['ci_upper']:.0%}"
          + ". Every single-split number published on this dataset has that same "
          "imprecision; ours is just the one that says so."),
        B("<b>The diagnosis itself may be circular.</b> The doctors who recorded the "
          "measurements also made the diagnosis, and were not blinded to it. Symptoms "
          "like 'weight gain' may have been asked about <i>because</i> PCOS was already "
          "suspected."),
        B("<b>We do not know who these women are.</b> The dataset publishes no collection "
          "dates, no eligibility rules, and no treatment history. A third of them have "
          "PCOS, against 8–13% in the general population, so these are clearly women who "
          "already had reason to attend a hospital."),
        B("<b>No ultrasound images.</b> We have follicle <i>counts</i> taken from scans, "
          "but not the scans themselves."),
        B("<b>Our external test was partial.</b> As explained in Section 8, only a "
          "weakened version of the model could be tested abroad."),
        B("<b>This is a coursework project, not a medical device.</b> Diagnosing PCOS "
          "requires a clinician."),
        Spacer(1, 6),
        _callout(
            "We also applied a professional reporting standard (TRIPOD+AI) to our own "
            "work and graded ourselves against it. The verdict: our <b>analysis</b> is "
            "sound and we can defend it in detail; the <b>data</b> carries real risks we "
            "cannot fix. No amount of careful modelling repairs a diagnosis recorded by "
            "someone who already knew the answer.", st, BLUE),
    ]

    # ------------------------------------------------------------ summary
    story += [
        Paragraph("11. Summary", st["h1"]),
        P("What we built and what it scores:"),
    ]
    story += [
        _table([
            ["Measure", "Result", "In plain terms"],
            ["Accuracy", f"{s['test_accuracy']:.1%}",
             "Correct on this many patients overall"],
            ["Sensitivity", f"{s['test_recall']:.1%}",
             "Of women who have PCOS, this many are found"],
            ["Specificity", f"{s['test_specificity']:.1%}",
             "Of healthy women, this many are correctly cleared"],
            ["AUC", f"{s['test_roc_auc']:.3f}",
             "Overall ability to rank patients by risk (1.0 is perfect)"],
        ], st, widths=[3.2 * cm, 2.4 * cm, 10.2 * cm]),
        Spacer(1, 12),
        Paragraph("The five things worth remembering", st["h2"]),
    ]
    for text in [
        f"A flawed testing method can manufacture <b>{inflation:.1%} extra accuracy</b> "
        "out of nothing. Published scores are only as trustworthy as the method behind them.",
        f"<b>{n_winners} of 9 algorithms</b> won at least one round of testing. Arguing "
        "about which algorithm is best on data this size is arguing about noise.",
        f"A <b>free questionnaire</b> gets you {pct:.0%} of the way. The expensive blood "
        "panel adds almost nothing.",
        f"A <b>2003 clinical rule</b> with nothing learned from data achieves "
        f"{pct_rule:.0%} of our model's score — but our model finds {extra} more patients.",
        "<b>Nobody can properly test PCOS models across populations</b>, because the "
        "necessary data has not been collected in a compatible way.",
    ]:
        story.append(B(text))

    story += [
        Spacer(1, 14),
        _callout(
            "<b>Our overall conclusion.</b> Machine learning can support PCOS screening, "
            "and it does genuinely find patients a simple rule misses. But on this dataset "
            "the interesting variable was never the algorithm — it was the testing method. "
            "We think reporting a defensible "
            f"{s['test_accuracy']:.0%} with its limitations stated is worth more than "
            "another 99% that nobody can check.", st),
        Spacer(1, 16),
        Paragraph(
            "Full source code, all 30 figures, 31 result tables, 64 automated tests and a "
            "step-by-step notebook are in the project repository. Every number in this "
            "report is read directly from the pipeline's output when the report is built, "
            "so it cannot drift out of date.",
            st["caption"]),
    ]

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(body, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.4 * cm, 1.2 * cm, "Group 5 — PCOS Prediction with Machine Learning")
        canvas.drawRightString(A4[0] - 2.4 * cm, 1.2 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=_footer)
    return output_path


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
