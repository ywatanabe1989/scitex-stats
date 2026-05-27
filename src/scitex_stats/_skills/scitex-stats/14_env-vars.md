---
description: |
  [TOPIC] Env Vars
  [DETAILS] see general/10_arch-environment-variables.md.
tags: [scitex-stats-env-vars, scitex-stats]
---


# scitex-stats — Environment Variables

| Variable | Purpose | Default | Type |
|---|---|---|---|
| `SCITEX_LOGGING_AVAILABLE` | Cross-package feature flag set by scitex-logging when present; scitex-stats reads it to decide whether to attach the SciTeX logger vs stdlib logging. | unset | bool (presence) |
| `SCITEX_STATS_CONFIG` | Path to a YAML config file, overrides user-scope `~/.scitex/stats/config.yaml`. Documented in the CLI precedence chain. | unset | str (path) |

## Notes

- No other module-private `SCITEX_STATS_*` vars are defined at this time.

## Audit

```bash
grep -rhoE 'SCITEX_[A-Z0-9_]+' $HOME/proj/scitex-stats/src/ | sort -u
```
