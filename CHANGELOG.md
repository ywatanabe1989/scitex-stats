# Changelog

All notable changes to `scitex-stats` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.18] — 2026-05-12

### Fixed
- `_export_report_html` crashed on pandas ≥ 2.2 — `Styler.applymap`
  was removed; now uses `Styler.map` with a fallback.
- Six standalone-demo modules (`effect_sizes/_cliffs_delta`,
  `_cohens_d`, `_epsilon_squared`, `_eta_squared`,
  `_prob_superiority`, `power/_power`) raised `NameError` on `stx`
  because `run_main()` referenced scitex without importing it.
- Seven more in-source demos (anova, anova_2way, anova_rm, shapiro,
  ks_1samp, ks_2samp, kruskal, mannwhitneyu, dunnett, games_howell,
  tukey_hsd) called the unsupported
  `convert_results(return_as="excel" / "csv")` and crashed; replaced
  with pandas `.to_excel` / `.to_csv` direct calls.
- Cleaned up a dead duplicate `_correct_fdr_.py` (and its bound
  demo) — the package always imported from `_correct_fdr` (no
  trailing underscore).
- `_figrecipe_integration.annotate` over-unwrapped `RecordingAxes`
  to raw `matplotlib.Axes`, which broke every direct figrecipe
  caller (`add_stat_annotation` lives on `RecordingAxes`).

### Changed
- Refactored `_utils/_normalizers.py` (927 LOC) into three focused
  modules — `_normalize_core`, `_export_files`, `_export_reports` —
  with the original kept as a thin re-exporting orchestrator (~58
  LOC). Public API unchanged.
- Extracted runnable demos out of `_correct_fdr`,
  `_correct_bonferroni`, `_correct_sidak`, `_correct_holm` into
  sibling `_demo_*.py` files. Each core file now fits the
  512-LOC project budget.
- README mirrors scitex-io's structure (self-contained Quick Start,
  numbered `## How it works` with mermaid, mermaid Available Tests
  + decision flowchart). README badge + `codecov.yml` now pin
  develop as the canonical branch.
- Examples converted from `.py` scripts to executed `.ipynb`
  notebooks; CI runs them end-to-end via `jupyter nbconvert
  --execute`.

### Added
- 23-module demo smoke test (`tests/integration/test_demos.py`)
  drives every `_demo_*.py` and `__main__`-bearing `_test_*.py`
  module end-to-end via `python -m …` in tmp_path.
- Direct unit tests for `_plot_anova_2way`, `_plot_holm`,
  `_plot_bonferroni`, `_plot_sidak`, `_plot_fdr`, `_decision_tree`
  render helpers, `_dispatch` branches, every `_mcp/_handlers/*`
  module, every `_cli/*` click subcommand, the `_server.py` FastMCP
  tools, and numpy-path tests for `_circular`/`_nan`/`_real`.
- Subprocess coverage tracking via `tests/conftest.py`
  (`COVERAGE_PROCESS_START` + `COVERAGE_FILE` pin + idempotent
  `.pth` shim) so demo subprocesses contribute coverage.
- `codecov.yml` with `branch: develop` + auto-target gates.
- `.scitex/dev/config.yaml` whitelisting `codecov.yml` and `logs/`.

### Coverage
- Project Codecov: **~32 % → ~90 %** without `omit` shortcuts.

## [0.2.17]

- Initial CHANGELOG entry — see git log for prior history.
