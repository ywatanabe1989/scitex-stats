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

## Notes

- scitex-stats defines no module-private `SCITEX_STATS_*` vars yet.
- The only env reference is an optional-dep probe; no user-tunable knobs at this time.

## Audit

```bash
grep -rhoE 'SCITEX_[A-Z0-9_]+' $HOME/proj/scitex-stats/src/ | sort -u
```
