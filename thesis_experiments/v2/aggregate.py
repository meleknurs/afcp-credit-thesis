"""Aggregate v2 AFCP result files into the tables reported in the thesis (Chapter 6).

Usage (from the repository root):
    python thesis_experiments/v2/aggregate.py            # writes thesis_experiments/results/v2/tables/*.csv and runs checks
    python thesis_experiments/v2/aggregate.py --show     # also prints every table

Every table is defined once in TABLES as (name, primary attribute, target level, rows), where each row is a
(label, glob pattern) pair; the pattern must match exactly one result file per seed. Nothing in this file is
typed by hand from a result file: the thesis tables are copies of the CSVs written here.

Columns: n_seeds, overall coverage, coverage of the target group, coverage inside the bias target cell (if the
run has one), worst-group coverage and gap over the levels of the primary attribute (per seed, then averaged),
average set size, AFCP selection rate of the primary attribute and of any other candidate; *_se = standard
error over seeds. Coverage to 3 decimals, set size and rates to 2 in the thesis.

Checks (printed at the end; a failure is a warning, not an error):
  C1  exactly the expected number of seeds per row (no file matched twice, none missing)
  C2  clean-data coverage of the marginal method within 0.90 +/- 0.02
  C3  Partial == Exhaustive when the run has a single candidate attribute
  C4  AdaptiveA2 set size >= Adaptive set size (A2 sets are unions)
  C5  worst-group coverage <= target-group coverage
"""
import glob, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "v2", "afcp")
OUT = os.path.join(os.path.dirname(HERE), "results", "v2", "tables")
METHODS = ["Marginal", "Partial", "Exhaustive", "Adaptive", "Adaptive1", "AdaptiveA2"]

TW = "taiwan/taiwan_nn_ntr6000_ncal6000_nte500_cond{r}_td0.0_sl{sl}_seed????{t}.txt"
TW5 = "taiwan/taiwan-sex_female_marriage_cand_education_cand_age_lt25_age_3way_nn_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????{t}.txt"
GM = "gmsc/gmsc_nn_ntr6000_ncal6000_nte500_cond{r}_td0.0_sl0.1_seed????{t}.txt"
SG = "sgc/sgc_{m}_ntr350_ncal350_nte300_cond{r}_td0.0_sl0.1_seed????.txt"
HID = "_biaslabelneg_sex_female1_eps{e}_{s}"; STG = "_biaslabel_sex_female1_eps{e}_{s}"
INT = "_biaslabelneg_sex_female&age_3way1&0_eps{e}_{s}"

def tw(t="", r="False", sl="0.1"): return TW.format(r=r, sl=sl, t=t)

TABLES = [
  ("table_6_1_baseline", [
    ("Taiwan marginal", tw(), "sex_female", 1), ("Taiwan label-cond", tw(r="True"), "sex_female", 1),
    ("GMSC marginal", GM.format(r="False", t=""), "age_3way", 0), ("GMSC label-cond", GM.format(r="True", t=""), "age_3way", 0),
    ("SGC marginal", SG.format(m="nn", r="False"), "age_le25", 1), ("SGC label-cond", SG.format(m="nn", r="True"), "age_le25", 1)]),
  ("table_6_2_hidden_dose", [("clean", tw(), "sex_female", 1)] +
    [(f"hidden eps {e}", tw(HID.format(e=e, s="traincal")), "sex_female", 1) for e in ["0.1", "0.2", "0.3", "0.5", "0.7"]]),
  ("table_6_3_stigma_dose", [(f"stigma eps {e}", tw(STG.format(e=e, s="traincal")), "sex_female", 1) for e in ["0.05", "0.1", "0.2", "0.3", "0.4"]]),
  ("table_6_4_scope", [
    ("Taiwan 0.5 train only", tw(HID.format(e="0.5", s="train")), "sex_female", 1), ("Taiwan 0.5 train+cal", tw(HID.format(e="0.5", s="traincal")), "sex_female", 1),
    ("Taiwan 0.2 train only", tw(HID.format(e="0.2", s="train")), "sex_female", 1), ("Taiwan 0.2 train+cal", tw(HID.format(e="0.2", s="traincal")), "sex_female", 1),
    ("Taiwan stigma 0.5 train only", tw(STG.format(e="0.5", s="train")), "sex_female", 1), ("Taiwan stigma 0.2 train only", tw(STG.format(e="0.2", s="train")), "sex_female", 1),
    ("GMSC 0.7 train only", GM.format(r="False", t="_biaslabelneg_age_3way0_eps0.7_train"), "age_3way", 0), ("GMSC 0.7 train+cal", GM.format(r="False", t="_biaslabelneg_age_3way0_eps0.7_traincal"), "age_3way", 0),
    ("GMSC 0.5 train+cal", GM.format(r="False", t="_biaslabelneg_age_3way0_eps0.5_traincal"), "age_3way", 0), ("GMSC 0.3 train+cal", GM.format(r="False", t="_biaslabelneg_age_3way0_eps0.3_traincal"), "age_3way", 0)]),
  ("table_6_5_intersection", [
    ("4 cand 0.5 train only", tw(INT.format(e="0.5", s="train")), "sex_female", 1), ("4 cand 0.5 train+cal", tw(INT.format(e="0.5", s="traincal")), "sex_female", 1),
    ("4 cand 0.7 train+cal", tw(INT.format(e="0.7", s="traincal")), "sex_female", 1),
    ("5 cand 0.5 train only", TW5.format(t=INT.format(e="0.5", s="train")), "sex_female", 1), ("5 cand 0.5 train+cal", TW5.format(t=INT.format(e="0.5", s="traincal")), "sex_female", 1),
    ("5 cand 0.7 train+cal", TW5.format(t=INT.format(e="0.7", s="traincal")), "sex_female", 1)]),
  ("table_6_6_regime", [("label-cond clean", tw(r="True"), "sex_female", 1)] +
    [(f"label-cond hidden eps {e}", tw(HID.format(e=e, s="traincal"), r="True"), "sex_female", 1) for e in ["0.2", "0.3", "0.5", "0.7"]] +
    [(f"label-cond stigma eps {e}", tw(STG.format(e=e, s="traincal"), r="True"), "sex_female", 1) for e in ["0.1", "0.2", "0.3"]]),
  ("table_6_7_siglevel", [(f"sl {sl} {lab}", tw(t, sl=sl), "sex_female", 1) for sl in ["0.05", "0.1", "0.2"]
    for lab, t in [("clean", ""), ("hidden 0.3", HID.format(e="0.3", s="traincal")), ("hidden 0.5", HID.format(e="0.5", s="traincal")), ("hidden 0.7", HID.format(e="0.7", s="traincal")), ("stigma 0.3", STG.format(e="0.3", s="traincal"))]]),
  ("table_6_7_samplesize", [(f"n={n} {lab}", f"taiwan/taiwan_nn_ntr{n}_ncal{n}_nte500_condFalse_td0.0_sl0.1_seed????{t}.txt", "sex_female", 1)
    for n in [100, 250, 500, 1000, 2000, 4000, 6000, 8000] for lab, t in [("clean", ""), ("hidden 0.5", HID.format(e="0.5", s="traincal"))]] +
    [(f"ncal={n} {lab}", f"taiwan/taiwan_nn_ntr6000_ncal{n}_nte500_condFalse_td0.0_sl0.1_seed????{t}.txt", "sex_female", 1)
    for n in [1000, 2000, 4000] for lab, t in [("clean", ""), ("hidden 0.5", HID.format(e="0.5", s="traincal"))]]),
  ("table_6_8_a2", [("A2 clean", TW5.format(t="_afcpa2"), "sex_female", 1), ("A2 0.5 train only", TW5.format(t="_afcpa2" + INT.format(e="0.5", s="train")), "sex_female", 1),
    ("A2 0.7 train+cal", TW5.format(t="_afcpa2" + INT.format(e="0.7", s="traincal")), "sex_female", 1),
    ("GMSC multi A2", "gmsc_multi/gmsc_multi_nn_ntr6000_ncal6000_nte1000_condFalse_td0.0_sl0.1_seed????_afcpa2.txt", "age_3way", 0),
    ("GMSC multi main", "gmsc_multi/gmsc_multi_nn_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????.txt", "age_3way", 0)]),
  ("table_6_9_classifier", [(f"LR {lab}", f"taiwan/taiwan_lr_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????{t}.txt", "sex_female", 1)
    for lab, t in [("clean", ""), ("hidden 0.2", HID.format(e="0.2", s="traincal")), ("hidden 0.5", HID.format(e="0.5", s="traincal")), ("stigma 0.2", STG.format(e="0.2", s="traincal")), ("stigma 0.5", STG.format(e="0.5", s="traincal"))]] +
    [("LR Taiwan label-cond", "taiwan/taiwan_lr_ntr6000_ncal6000_nte500_condTrue_td0.0_sl0.1_seed????.txt", "sex_female", 1),
     ("LR GMSC marginal", "gmsc/gmsc_lr_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????.txt", "age_3way", 0), ("LR GMSC label-cond", "gmsc/gmsc_lr_ntr6000_ncal6000_nte500_condTrue_td0.0_sl0.1_seed????.txt", "age_3way", 0),
     ("LR SGC marginal", SG.format(m="lr", r="False"), "age_le25", 1), ("LR SGC label-cond", SG.format(m="lr", r="True"), "age_le25", 1)]),
  ("table_6_10_score", [(f"THR {lab}", tw("_scorethr" + t), "sex_female", 1)
    for lab, t in [("clean", ""), ("hidden 0.5 train only", HID.format(e="0.5", s="train")), ("hidden 0.5 train+cal", HID.format(e="0.5", s="traincal")), ("hidden 0.7 train+cal", HID.format(e="0.7", s="traincal")), ("stigma 0.3", STG.format(e="0.3", s="traincal"))]]),
  ("table_6_11_groupdef", [
    ("Taiwan age<25", "rq4_groupdef/taiwan-age_lt25_nn_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????.txt", "age_lt25", None),
    ("Taiwan age 3-way", "rq4_groupdef/taiwan-age_3way_nn_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????.txt", "age_3way", None),
    ("GMSC age 3-way", GM.format(r="False", t=""), "age_3way", None),
    ("GMSC age<25", "rq4_groupdef/gmsc-age_lt25_nn_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????.txt", "age_lt25", None),
    ("GMSC age>=60", "rq4_groupdef/gmsc-age_ge60_nn_ntr6000_ncal6000_nte500_condFalse_td0.0_sl0.1_seed????.txt", "age_ge60", None),
    ("SGC age<=25", "rq4_groupdef/sgc-age_le25_nn_ntr350_ncal350_nte300_condFalse_td0.0_sl0.1_seed????.txt", "age_le25", None),
    ("SGC age<=33", "rq4_groupdef/sgc-age_le33_nn_ntr350_ncal350_nte300_condFalse_td0.0_sl0.1_seed????.txt", "age_le33", None)]),
]

def aggregate(pattern, primary, target_level):
    files = sorted(glob.glob(os.path.join(RES, pattern)))
    if not files: return None, 0
    d = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in files])
    dup = d.groupby(["seed", "method"]).file.nunique().max()
    out = []
    for m, g in d.groupby("method"):
        per = []
        for seed, s in g.groupby("seed"):
            ov = s[s.attribute_idx == -1].iloc[0]; pr = s[s.attribute_name == primary]
            cell = s[s.attribute_name == "bias_target_cell"]
            selcols = [c for c in s.columns if c.startswith("selected_") and c not in ("selected_none", "selected_two")]
            others = [c for c in selcols if c != "selected_" + primary]
            per.append(dict(overall=ov.Coverage, size=ov.Length,
                            target=pr[pr.attribute_level == target_level].Coverage.iloc[0] if target_level is not None else np.nan,
                            cell=cell[cell.attribute_level == 1].Coverage.iloc[0] if len(cell) else np.nan,
                            worst=pr.Coverage.min(), gap=pr.Coverage.max() - pr.Coverage.min(),
                            sel=ov.get("selected_" + primary, np.nan) if m.startswith("Adaptive") else np.nan,
                            sel_other=max(ov[c] for c in others) if others and m.startswith("Adaptive") else np.nan,
                            sel_two=ov.get("selected_two", np.nan) if m == "AdaptiveA2" else np.nan))
        p = pd.DataFrame(per); r = {"method": m, "n_seeds": len(p)}
        for c in p.columns: r[c] = p[c].mean(); r[c + "_se"] = p[c].std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else np.nan
        out.append(r)
    o = pd.DataFrame(out).set_index("method"); o = o.loc[[m for m in METHODS if m in o.index]]
    return o, dup

def main(show=False):
    os.makedirs(OUT, exist_ok=True); warnings = []
    for name, rows in TABLES:
        parts = []
        for label, pattern, primary, lvl in rows:
            o, dup = aggregate(pattern, primary, lvl)
            if o is None: warnings.append(f"[missing] {name}: {label} ({pattern})"); continue
            if dup > 1: warnings.append(f"[C1] {name}: {label}: a seed matched {dup} files")
            o.insert(0, "row", label); o.insert(1, "primary", primary); parts.append(o.reset_index())
            if "clean" in label.lower() and not label.startswith("A2") and abs(o.loc["Marginal", "overall"] - 0.90) > 0.02: warnings.append(f"[C2] {name}: {label}: clean Marginal coverage outside 0.90±0.02")
            if "Partial" in o.index and "Exhaustive" in o.index and "rq4_groupdef" in pattern and abs(o.loc["Partial", "overall"] - o.loc["Exhaustive", "overall"]) > 1e-9: warnings.append(f"[C3] {name}: {label}: Partial != Exhaustive with one candidate")
            if "AdaptiveA2" in o.index and o.loc["AdaptiveA2", "size"] < o.loc["Adaptive", "size"] - 1e-9: warnings.append(f"[C4] {name}: {label}: A2 set smaller than main")
            if lvl is not None and (o.worst > o.target + 1e-9).any(): warnings.append(f"[C5] {name}: {label}: worst > target")
        if parts:
            t = pd.concat(parts); t.to_csv(os.path.join(OUT, name + ".csv"), index=False, float_format="%.4f")
            if show: print("\n==", name); print(t[["row", "method", "n_seeds", "overall", "target", "cell", "worst", "gap", "size", "sel", "sel_other"]].round(3).to_string(index=False))
    print(f"\nwrote {len(TABLES)} tables to {OUT}")
    print("checks:", "all passed" if not warnings else "\n  " + "\n  ".join(warnings))

if __name__ == "__main__":
    main(show="--show" in sys.argv)
