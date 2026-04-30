---
name: scitex-stats-env-vars
description: Environment variables read by scitex-stats at import / runtime. Follow SCITEX_<MODULE>_* convention — see general/10_arch-environment-variables.md.
tags: [scitex-stats, scitex-package]
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
