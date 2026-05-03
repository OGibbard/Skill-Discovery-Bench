# Publication Checklist

Use this when preparing this folder for GitHub upload.

## Recommended Upload Path

This folder is intended to be uploaded as the repository root. The existing `.git` history comes from the larger dissertation workspace, so create a fresh repository from this cleaned folder rather than pushing the old workspace history.

Recommended fresh-repo flow:

```bash
cd /path/to/unified-skill-discovery
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin git@github.com:OGibbard/unified-skill-discovery.git
git push -u origin main
```

## Before Pushing

- Confirm `LICENSE` and `THIRD_PARTY_NOTICES.md` are present.
- Confirm `rllab/config_personal.py` is not committed; use `rllab/config_personal_template.py` for local/cloud settings.
- Confirm no `outputs/`, `.pkl`, videos, `.DS_Store`, `__pycache__`, course PDFs, or copied papers are staged.
- Run the compile check used by GitHub Actions:

```bash
python -m compileall -q \
  run_full_diayn.py \
  run_scaled_diayn.py \
  mujoco_diayn_sim.py \
  visualize_diayn.py \
  analysis \
  sac \
  rllab
```

- Keep the GitHub description explicit: "Unofficial modernized same-stack DIAYN/DADS/LSD robustness code."
- Add GitHub topics such as `reinforcement-learning`, `skill-discovery`, `diayn`, `dads`, `lsd`, `sac`, `mujoco`, and `robustness`.
- Tag the first public release as `v0.1.0`.
