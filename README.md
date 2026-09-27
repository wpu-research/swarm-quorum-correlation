# Shared weights, shared scenes — code and results

Code, result files and figures for the manuscript

> M. Arslan, *Shared Weights, Shared Scenes: Measuring Viewpoint, Condition and Detector Components of Error Correlation for Quorum Decisions in Vision-Based UAV Swarms* (submitted to *Drones*, MDPI, 2026).

A quorum rule ("commit when at least Q of N agents confirm") assumes that agents err independently. When every UAV in a swarm runs the same detector, it doesn't. This repository measures how correlated the agents' errors are and what that does to swarm decisions. The correlation is split into three components: viewpoint (bearing separation), shared scene condition (illumination and background), and detector. The measured kernels then feed a Monte-Carlo simulation of quorum rules.

## What is included

| Path | Content |
|---|---|
| `sim/swarm/ispat/d1_klip.py` | Deterministic synthetic stimulus generator (frames are regenerated from seeds; they are never stored) |
| `sim/swarm/ispat/d1_coklu.py` | Multi-detector runner: renders every frame once and feeds it to M detectors; shared or independent degradation |
| `sim/swarm/ispat/d1_revizyon.py` | Estimator: count-based vectorised tetrachoric correlation, pooling levels, stratum bootstrap, kernel (Eq. 3) |
| `sim/swarm/ispat/d1_revizyon_kos.py` | Full measurement analysis → `sonuclar_rev/revizyon*.json` |
| `sim/swarm/ispat/d1_revizyon_sim.py` | Simulation core: measured kernels, decision rules compared on the ROC lower convex hull at an exact miss budget |
| `sim/swarm/ispat/d1_revizyon_sim_kos.py` | Simulations → `sonuclar_rev/simulasyon*.json` |
| `sim/swarm/ispat/sonuclar_rev/` | All measured and simulated quantities that appear in the paper (JSON) |
| `docs/swarm/makale/sekiller_rev.py` | Figure generator, which reads only the JSON files |
| `docs/swarm/makale/sekiller_rev/` | The figures |
| `d1/klip_ayar_tam70.yaml` | Stimulus and degradation configuration used for all runs |

Code comments are partly in Turkish; function names and docstrings describe each step.

## What is not included

The object view sheets used as stimuli were generated with a text-to-image model. They and the per-frame detection records (about 8 MB) are **not** included; they are available from the author on reasonable request. Without them you can:

- regenerate every **simulation** and **figure** from the JSON files, but
- not re-run the **measurement** stage (`d1_coklu.py`, `d1_revizyon_kos.py`).

The detector weights are the public Ultralytics COCO models plus one YOLO11n fine-tuned on a public UAV/bird dataset; the fine-tuned weights are not included.

## Reproduce

```bash
pip install -r requirements.txt
cd sim/swarm/ispat
MC=400000 python3 d1_revizyon_sim_kos.py                      # shared degradation → sonuclar_rev/simulasyon.json
REV=revizyon_bagimsiz.json SIM=simulasyon_bagimsiz.json \
  MC=400000 python3 d1_revizyon_sim_kos.py                    # independent degradation
cd ../../../docs/swarm/makale && python3 sekiller_rev.py      # → sekiller_rev/*.png
```

All seeds are fixed. With the JSON files in this repository the figure script reproduces the published figures byte for byte.

## Licence

- Code: MIT (`LICENSE`).
- Result files and figures (`sonuclar_rev/`, `sekiller_rev/`): CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. A DOI will be minted through Zenodo for each release.
