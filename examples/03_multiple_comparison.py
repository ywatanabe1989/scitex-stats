#!/usr/bin/env python3
# File: examples/03_multiple_comparison.py
"""Multiple comparison correction example."""

from pathlib import Path

from scitex_stats import correct

OUT_DIR = Path(__file__).parent / "03_multiple_comparison_out"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Simulate p-values from multiple tests
    # correct_fdr expects a list of dicts with 'pvalue' keys
    results = [
        {"pvalue": 0.01, "var_x": "A", "var_y": "B"},
        {"pvalue": 0.04, "var_x": "A", "var_y": "C"},
        {"pvalue": 0.03, "var_x": "A", "var_y": "D"},
        {"pvalue": 0.20, "var_x": "B", "var_y": "C"},
        {"pvalue": 0.005, "var_x": "B", "var_y": "D"},
        {"pvalue": 0.08, "var_x": "C", "var_y": "D"},
    ]

    # Apply FDR correction (Benjamini-Hochberg)
    corrected = correct.correct_fdr(results, alpha=0.05, method="bh", verbose=False)

    lines = [
        "Multiple Comparison Correction (FDR-BH)",
        "=" * 50,
        f"{'Comparison':>10} {'Original':>10} {'Adjusted':>10} {'Rejected':>10}",
        "-" * 45,
    ]
    for orig, adj in zip(results, corrected):
        label = f"{orig['var_x']}v{orig['var_y']}"
        sig = "Yes" if adj["rejected"] else "No"
        lines.append(
            f"{label:>10} {orig['pvalue']:>10.4f} {adj['pvalue_adjusted']:>10.4f} {sig:>10}"
        )
    output = "\n".join(lines)
    print(output)

    (OUT_DIR / "results.txt").write_text(output + "\n")

    # Save as JSON
    import json

    json_data = {
        "method": "FDR (Benjamini-Hochberg)",
        "alpha": 0.05,
        "results": [
            {
                "comparison": f"{orig['var_x']}v{orig['var_y']}",
                "pvalue_original": orig["pvalue"],
                "pvalue_adjusted": adj["pvalue_adjusted"],
                "rejected": adj["rejected"],
            }
            for orig, adj in zip(results, corrected)
        ],
    }
    (OUT_DIR / "results.json").write_text(json.dumps(json_data, indent=2) + "\n")

    print(f"\nSaved to {OUT_DIR}/")
    print("  results.txt  — human-readable summary")
    print("  results.json — structured data")


if __name__ == "__main__":
    main()
