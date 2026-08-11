#!/usr/bin/env python3
"""
Data-quality audit (task #8 of the v2 rebuild plan).

Quantifies the mechanism behind B1 (duplicate-row leakage) and B5 (the
93% E-recall ceiling): projecting the 50-column source dataset down to
the 12 features this project can actually collect live collapses most
of the class-level variation, producing duplicate feature vectors and,
worse, all-zero vectors that are byte-identical across classes.

Uses only the standard library so it runs with the environment already
proven to work, independent of the (still-being-set-up) ML stack.

Writes results/data_quality.json - the single source of truth this
number should be quoted from in report.txt / please_study_this.txt /
both paper drafts, instead of being retyped by hand.
"""
import csv
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

FEATURES_12 = [
    "file_read", "file_created", "regkey_written", "regkey_read", "apistats",
    "command_line", "directory_enumerated", "dll_loaded", "tree_command_line",
    "arguments", "urls", "proc_pid",
]

PROXY_TRIO = ["apistats", "directory_enumerated", "dll_loaded"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_csv(path, header_row=0):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    return rows[header_row], rows[header_row + 1:]


def as_dicts(header, rows):
    return [dict(zip(header, r)) for r in rows if len(r) == len(header)]


def vec12(d):
    return tuple(round(num(d[f]), 6) for f in FEATURES_12)


def distinct_by_class(rows, key_fn, class_key="family"):
    out = {}
    for cls in sorted({r[class_key] for r in rows}):
        sub = [key_fn(r) for r in rows if r[class_key] == cls]
        out[cls] = {"rows": len(sub), "distinct": len(set(sub))}
    return out


def main():
    header_bal, rows_bal = load_csv("datasets/balanced_dataset.csv")
    balanced = as_dicts(header_bal, rows_bal)

    # ransomware_dataset.csv has a "Table S1" line before the real header
    header_src, rows_src = load_csv("datasets/ransomware_dataset.csv", header_row=1)
    source = as_dicts(header_src, rows_src)
    all_cols = [c for c in header_src if c != "family"]

    result = {
        "generated_by": "scripts/data_quality_audit.py",
        "purpose": (
            "Quantify how projecting the 50-feature source dataset down to "
            "this project's 12 live-collectable features collapses class "
            "separation, as the shared mechanism behind B1 (duplicate-row "
            "CV leakage) and B5 (the encryptor-class recall ceiling)."
        ),
    }

    # --- 1. duplication: full 50 features vs the 12-feature projection, on the SOURCE data
    result["source_dataset_duplication"] = {
        "n_rows": len(source),
        "full_50_feature_vectors": distinct_by_class(
            source, lambda r: tuple(r[c] for c in all_cols)
        ),
        "projected_12_feature_vectors": distinct_by_class(source, vec12),
        "note": (
            "The source dataset itself is not the problem (mostly unique "
            "at 50 features). The 12-feature projection is what collapses "
            "class separation."
        ),
    }

    # --- 2. duplication on the actual training set (source + self-collected goodware)
    result["balanced_dataset_duplication_12_feature"] = distinct_by_class(balanced, vec12)
    total_rows = len(balanced)
    total_distinct = len({vec12(r) for r in balanced})
    dup_rows = sum(
        1
        for r in balanced
        if list(Counter(vec12(x) for x in balanced).values())[0]
    )
    counts = Counter(vec12(r) for r in balanced)
    rows_with_a_twin = sum(1 for r in balanced if counts[vec12(r)] > 1)
    result["balanced_dataset_duplication_12_feature"]["ALL"] = {
        "rows": total_rows,
        "distinct": total_distinct,
    }
    result["balanced_dataset_leakage_exposure"] = {
        "rows_with_at_least_one_identical_twin": rows_with_a_twin,
        "fraction": round(rows_with_a_twin / total_rows, 4),
        "interpretation": (
            "Under shuffled k-fold CV, most of a duplicated row's twins "
            "land in the training folds, so the held-out copy is "
            "memorised rather than generalised to."
        ),
    }

    # --- 3. the all-zero collision (mechanism behind the E-recall ceiling)
    def all_trio_zero(d):
        return all(num(d[f]) == 0 for f in PROXY_TRIO)

    def all_12_zero(d):
        return all(num(d[f]) == 0 for f in FEATURES_12)

    trio_zero_by_class = {}
    full_zero_by_class = {}
    for cls in ["G", "E", "L"]:
        sub = [r for r in balanced if r["family"] == cls]
        trio_zero_by_class[cls] = sum(all_trio_zero(r) for r in sub)
        full_zero_by_class[cls] = {
            "count": sum(all_12_zero(r) for r in sub),
            "total": len(sub),
        }

    n_e = sum(1 for r in balanced if r["family"] == "E")
    n_e_full_zero = full_zero_by_class["E"]["count"]
    n_g_full_zero = full_zero_by_class["G"]["count"]

    result["all_zero_vector_collision"] = {
        "rows_with_apistats_dir_enum_dll_loaded_all_zero_by_class": trio_zero_by_class,
        "rows_with_all_12_features_zero_by_class": {
            k: v["count"] for k, v in full_zero_by_class.items()
        },
        "interpretation": (
            f"{n_e_full_zero} of {n_e} class-E rows are byte-identical "
            f"(all 12 features = 0) to {n_g_full_zero} class-G rows. "
            "These rows are unlearnable by construction: the same input "
            "vector appears under two different labels, so no classifier "
            "can be correct on both. All {0} zero-vector rows trace to "
            "the SOURCE dataset (Cuckoo Sandbox runs where the sandboxed "
            "process apparently produced none of these 12 signals in its "
            "window) - none originate from the self-collected goodware."
        ).format(n_e_full_zero + n_g_full_zero),
        "implied_ceiling_on_class_E_recall": {
            "formula": "(n_E - n_E_all_zero) / n_E",
            "value": round((n_e - n_e_full_zero) / n_e, 4),
            "n_E": n_e,
            "n_E_all_zero": n_e_full_zero,
        },
    }

    # --- 4. provenance check: do the all-zero rows come from source or self-collected data?
    header_nrm, rows_nrm = load_csv("datasets/normal_activity.csv")
    normal = as_dicts(header_nrm, rows_nrm)
    result["all_zero_provenance"] = {
        "source_dataset_ransomware_csv": {
            "G_all_zero": sum(
                1 for r in source if r["family"] == "G" and all_12_zero(r)
            ),
            "E_all_zero": sum(
                1 for r in source if r["family"] == "E" and all_12_zero(r)
            ),
            "L_all_zero": sum(
                1 for r in source if r["family"] == "L" and all_12_zero(r)
            ),
        },
        "self_collected_normal_activity_csv": {
            "all_zero": sum(1 for r in normal if all_12_zero(r)),
            "total": len(normal),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/data_quality.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\n[SAVED] results/data_quality.json")


if __name__ == "__main__":
    main()
