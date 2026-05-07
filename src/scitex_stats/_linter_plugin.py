"""Linter plugin for scitex-stats: statistics-specific rules (ST001-ST006).

Registered via entry point 'scitex_linter.plugins' so scitex-linter
discovers these rules automatically when scitex-stats is installed.
"""


def get_plugin():
    """Return scitex-stats linter rules and call mappings."""
    from scitex_dev.linter._rules._base import Rule

    ST001 = Rule(
        id="STX-ST001",
        severity="warning",
        category="stats",
        message="`scipy.stats.ttest_ind()` — use `stx.stats.ttest_ind()` for auto effect size + CI",
        suggestion="Replace with `stx.stats.ttest_ind(a, b)` which includes Cohen's d and CI.",
        requires="scitex",
    )

    ST002 = Rule(
        id="STX-ST002",
        severity="warning",
        category="stats",
        message="`scipy.stats.mannwhitneyu()` — use `stx.stats.mannwhitneyu()` for auto effect size",
        suggestion="Replace with `stx.stats.mannwhitneyu(a, b)` which includes Cliff's delta.",
        requires="scitex",
    )

    ST003 = Rule(
        id="STX-ST003",
        severity="warning",
        category="stats",
        message="`scipy.stats.pearsonr()` — use `stx.stats.pearsonr()` for auto CI + power",
        suggestion="Replace with `stx.stats.pearsonr(a, b)` which includes CI and power analysis.",
        requires="scitex",
    )

    ST004 = Rule(
        id="STX-ST004",
        severity="warning",
        category="stats",
        message="`scipy.stats.f_oneway()` — use `stx.stats.anova_oneway()` for post-hoc + effect sizes",
        suggestion="Replace with `stx.stats.anova_oneway(*groups)` which includes eta-squared.",
        requires="scitex",
    )

    ST005 = Rule(
        id="STX-ST005",
        severity="warning",
        category="stats",
        message="`scipy.stats.wilcoxon()` — use `stx.stats.wilcoxon()` for auto effect size",
        suggestion="Replace with `stx.stats.wilcoxon(a, b)` which includes effect size and CI.",
        requires="scitex",
    )

    ST006 = Rule(
        id="STX-ST006",
        severity="warning",
        category="stats",
        message="`scipy.stats.kruskal()` — use `stx.stats.kruskal()` for post-hoc + effect sizes",
        suggestion="Replace with `stx.stats.kruskal(*groups)` which includes epsilon-squared.",
        requires="scitex",
    )

    return {
        "rules": [ST001, ST002, ST003, ST004, ST005, ST006],
        "call_rules": {
            ("stats", "ttest_ind"): ST001,
            ("scipy.stats", "ttest_ind"): ST001,
            ("stats", "mannwhitneyu"): ST002,
            ("scipy.stats", "mannwhitneyu"): ST002,
            ("stats", "pearsonr"): ST003,
            ("scipy.stats", "pearsonr"): ST003,
            ("stats", "f_oneway"): ST004,
            ("scipy.stats", "f_oneway"): ST004,
            ("stats", "wilcoxon"): ST005,
            ("scipy.stats", "wilcoxon"): ST005,
            ("stats", "kruskal"): ST006,
            ("scipy.stats", "kruskal"): ST006,
        },
        "axes_hints": {},
        "checkers": [],
    }
