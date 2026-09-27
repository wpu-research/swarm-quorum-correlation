#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revised-manuscript figures (English) — every number read from
sim/swarm/ispat/sonuclar_rev/{revizyon,simulasyon}.json; nothing hand-typed.

    python3 sekiller_rev.py      # → sekiller_rev/fig*.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "..", "sim", "swarm", "ispat", "sonuclar_rev")
OUT = os.path.join(HERE, "sekiller_rev"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(RES, "revizyon.json")))
S = json.load(open(os.path.join(RES, "simulasyon.json")))
DT = np.arange(0, 181, 15.0)

# categorical slots, fixed order (dataviz reference palette, light mode)
C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID = "#1f1f1e", "#5a5a55", "#d9d8d2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight"})

NAME = {"A": "YOLO11n (A, x86)", "B": "YOLOv8s-Hailo (B)", "yolo11n": "YOLO11n", "yolo11s": "YOLO11s",
        "yolo11m": "YOLO11m", "yolov8n": "YOLOv8n", "yolov8s": "YOLOv8s", "yolov5su": "YOLOv5su",
        "yolo26n": "YOLO26n", "yolov10s": "YOLOv10s", "rtdetr-l": "RT-DETR-l", "drone": "YOLO11n-drone",
        "yolo11n@640": "YOLO11n @640", "yolov8s@640": "YOLOv8s @640"}


def kernel(d, w, b, th, m):
    return w + b * (np.exp(-d / th) + m * np.exp(-(180 - d) / th))


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def band(ax, e, color, label, marker, key="egri", ga="egri_ga"):
    y = arr(e[key]); ax.plot(DT, y, marker=marker, ms=4, color=color, label=label)
    if ga in e:
        lo, hi = arr(e[ga][0]), arr(e[ga][1])
        ax.fill_between(DT, lo, hi, color=color, alpha=.15, lw=0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("→", name)


# ---------------------------------------------------------------- Fig 4: same-model decomposition
def fig_same():
    am = R["ayni_model"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    a = ax[0]
    for h, c, mk, lab in (("deneme", C3, "o", "within trial"), ("isik", C4, "s", "pooled over illumination"),
                          ("kosul", C1, "D", "pooled over illumination + background")):
        band(a, am["tuzak"][h]["A"], c, lab, mk)
    p = am["tuzak"]["kosul"]["A"]["fit"]; xx = np.linspace(1, 180, 200)
    a.plot(xx, kernel(xx, *p), color=INK, lw=1, ls="--", label="fitted kernel, Eq. (3)")
    a.set(title="(a) Detector A, decoys: pooling level", xlabel="bearing separation Δθ (°)",
          ylabel="error correlation ρ(Δθ)", xlim=(0, 185), ylim=(-.15, .8))
    a.legend(fontsize=6.5, loc="upper center")
    a = ax[1]
    for cl, c, mk, lab in (("kus", C1, "o", "birds"), ("direk", C2, "s", "poles")):
        band(a, am[cl]["kosul"]["A"], c, lab, mk)
    a.set(title="(b) Detector A by decoy class", xlabel="bearing separation Δθ (°)")
    a.legend(fontsize=6.5)
    a = ax[2]
    band(a, am["tuzak"]["kosul"]["A"], C1, "all view pairs", "D")
    a.plot(DT, arr(R["ayna_dislanmis"]["A"]["egri"]), marker="x", ms=5, color=C2, label="exact mirror/duplicate pairs removed")
    a.set(title="(c) Mirror-image view pairs", xlabel="bearing separation Δθ (°)")
    a.legend(fontsize=6.5)
    for a in ax:
        a.axhline(0, color=INK2, lw=.6)
    save(fig, "fig4_same_model.png")


# ---------------------------------------------------------------- Fig 5: cross-model
def fig_cross():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = ax[0]
    band(a, R["ayni_model"]["tuzak"]["kosul"]["A"], C1, "same model A–A", "D")
    band(a, R["ayni_model"]["tuzak"]["kosul"]["B"], C3, "same model B–B", "s")
    band(a, R["capraz"]["kosul"]["A|B"], C2, "cross model A–B", "o")
    a.axhline(0, color=INK2, lw=.6)
    a.set(title="(a) Same frames, two detectors (condition-pooled)", xlabel="bearing separation Δθ (°)",
          ylabel="error correlation ρ(Δθ)", xlim=(-5, 185), ylim=(-.15, .8))
    a.legend(fontsize=6.5)
    # heatmap of the 60–120° band (bearing-independent part), condition-pooled
    mods = [m for m in R["ayni_model"]["tuzak"]["kosul"] if m not in ("A",)]
    n = len(mods); M = np.full((n, n), np.nan)
    for i, a_ in enumerate(mods):
        M[i, i] = R["ayni_model"]["tuzak"]["kosul"][a_]["orta_bant"] or np.nan
        for j, b_ in enumerate(mods):
            if i < j:
                e = R["capraz"]["kosul"].get(f"{a_}|{b_}") or R["capraz"]["kosul"].get(f"{b_}|{a_}")
                if e and e["orta_bant"] is not None:
                    M[i, j] = M[j, i] = e["orta_bant"]
    a = ax[1]
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=max(.4, np.nanmax(M)))
    a.set_xticks(range(n)); a.set_yticks(range(n))
    a.set_xticklabels([NAME.get(m, m) for m in mods], rotation=60, ha="right", fontsize=6.5)
    a.set_yticklabels([NAME.get(m, m) for m in mods], fontsize=6.5); a.grid(False)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                       color="white" if M[i, j] > .25 else INK)
    a.set_title("(b) Bearing-independent correlation (Δθ = 60–120°)")
    fig.colorbar(im, ax=a, fraction=.046, pad=.02).set_label("ρ, condition-pooled")
    save(fig, "fig5_cross_model.png")


# ---------------------------------------------------------------- Fig 6: robustness
def fig_robust():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.1))
    a = ax[0]
    ab = R["ablation"]; mods = list(ab.get("kosul", {}))
    x = np.arange(len(mods)); w = .2
    for k, (h, key, c, lab) in enumerate((("deneme", "ortak_orta", C3, "within trial, shared degradation"),
                                           ("deneme", "bagimsiz_orta", C5, "within trial, independent degradation"),
                                           ("kosul", "ortak_orta", C1, "condition-pooled, shared"),
                                           ("kosul", "bagimsiz_orta", C2, "condition-pooled, independent"))):
        y = [ab[h][m][key] for m in mods]
        lo = [ab[h][m][key + "_ga"][0] for m in mods]; hi = [ab[h][m][key + "_ga"][1] for m in mods]
        a.bar(x + (k - 1.5) * w, y, w * .9, color=c, label=lab,
              yerr=[np.array(y) - lo, np.array(hi) - y], ecolor=INK2, capsize=1.5, error_kw=dict(lw=.6))
    a.set_xticks(x); a.set_xticklabels([NAME.get(m, m) for m in mods], fontsize=7)
    a.axhline(0, color=INK2, lw=.6)
    a.set(title="(a) Degradation ablation, Δθ = 60–120° band", ylabel="ρ")
    a.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    a = ax[1]
    ed = R.get("esik_degismezligi", {})
    mods2 = sorted({k.split("@")[0] for k in ed})
    for i, m in enumerate(mods2):
        th = [0.25, 0.40, 0.55]
        y = [ed[f"{m}@{t}"]["orta_bant"] for t in th]
        lo = [ed[f"{m}@{t}"]["orta_bant_ga"][0] for t in th]; hi = [ed[f"{m}@{t}"]["orta_bant_ga"][1] for t in th]
        a.errorbar(np.array(th) + (i - 1.5) * .008, y, yerr=[np.array(y) - lo, np.array(hi) - y],
                   marker="osD^"[i % 4], ms=5, color=(C1, C2, C3, C5)[i % 4], capsize=2, label=NAME.get(m, m))
    a.set(title="(b) Copula check: correlation vs confidence threshold", xlabel="confidence threshold",
          ylabel="ρ, Δθ = 60–120°, condition-pooled", xticks=[.25, .4, .55])
    a.legend(fontsize=6.5)
    save(fig, "fig6_robustness.png")


# ---------------------------------------------------------------- Fig 7: decision rules
def fig_rules():
    g_lab = {"esit": "evenly spread", "yarim": "180° arc", "iki_kume": "two clusters", "tek_kume": "single cluster"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
    for a, cfg, t in zip(ax, ("K1_A", "K8"), ("(a) one model (A) on every agent", "(b) eight distinct models")):
        gs = list(S["S4"]); x = np.arange(len(gs)); w = .26
        for k, (key, c, lab) in enumerate((("esik_butce", C1, "count threshold (randomised)"),
                                           ("neff_butce", C2, "diversity-weighted n_eff"),
                                           ("lr_butce", C3, "correlation-aware LR"))):
            a.bar(x + (k - 1) * w, [S["S4"][g][cfg][key] for g in gs], w * .9, color=c, label=lab)
        a.axhline(0.01, color=INK, ls="--", lw=1); a.text(x[-1] + .45, 0.0105, "1% target", fontsize=6.5, ha="right")
        a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([g_lab[g] for g in gs])
        a.set_title(t)
    ax[0].set_ylabel("P(false commit) at 20% miss budget"); ax[0].legend(fontsize=6.5, loc="upper left")
    save(fig, "fig7_rules.png")


# ---------------------------------------------------------------- Fig 8: diversity vs geometry
def fig_div():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
    for a, g, t in zip(ax, ("esit", "tek_kume"), ("(a) evenly spread", "(b) single cluster (±10°)")):
        d = S["S3"][g]
        for key, c, mk, lab, off in (("Q4", C1, "o", "Q = 4 of 8", -.08), ("esik_butce", C2, "s", "best threshold, 20% miss budget", .08)):
            for K, lst in d.items():
                y = [r[key][0] if key == "Q4" else r[key] for r in lst]
                xs = np.full(len(y), int(K)) * (1 + off * 0)
                a.scatter(np.log2(int(K)) + off + np.random.default_rng(int(K)).normal(0, .02, len(y)), y,
                          s=10, color=c, alpha=.5, marker=mk, lw=0)
            ks = sorted(d, key=int)
            a.plot([np.log2(int(k)) + off for k in ks],
                   [np.median([r[key][0] if key == "Q4" else r[key] for r in d[k]]) for k in ks],
                   color=c, marker=mk, ms=6, label=lab)
        a.axhline(0.01, color=INK, ls="--", lw=1)
        a.set_xticks(np.log2([1, 2, 4, 8])); a.set_xticklabels(["1", "2", "4", "8"])
        a.set(yscale="log", xlabel="number of distinct detectors K (N = 8 agents)", title=t)
    ax[0].set_ylabel("P(false commit)"); ax[0].legend(fontsize=6.5)
    save(fig, "fig8_diversity.png")


# ---------------------------------------------------------------- Fig 1: architecture
def fig_arch():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
    def box(x, y, w, h, t, c, fill="#ffffff", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", ec=c, fc=fill, lw=1.4))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5, color=INK, weight="bold" if bold else "normal")
    def arr(x0, y0, x1, y1, c=INK2):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1))
    ax.text(1, 34, "Per agent (runs on every UAV)", fontsize=8, color=C1, weight="bold")
    for i, t in enumerate(("Camera", "Detector\n(shared or\ndistinct weights)", "Confirmation\nc_i ∈ {0,1}")):
        box(2 + i * 17, 22, 14, 8, t, C1)
        if i: arr(2 + i * 17 - 3.6, 26, 2 + i * 17 - 0.4, 26)
    box(56, 22, 16, 8, "Safety envelope\n(enforced below the\nswarm layer)", C2, fill="#fdf0ea")
    ax.text(1, 16, "Swarm layer (this paper)", fontsize=8, color=C3, weight="bold")
    box(18, 3, 26, 9, "Commit decision\nn ≥ Q  ·  n_eff ≥ Q  ·  Λ ≥ λ", C3, fill="#e9f7f1", bold=True)
    box(52, 3, 18, 9, "Task allocation\n(engagement)", INK2)
    box(76, 3, 20, 9, "Search / next bearings", INK2)
    arr(42.5, 22, 34, 12.6, C3); arr(44.6, 7.5, 51.4, 7.5); arr(70.6, 7.5, 75.4, 7.5)
    ax.text(47, 17.5, "N confirmations, bearings Δθ_ij", fontsize=6.5, color=INK2)
    ax.text(20, 0, "Correlated perception (shared model, shared scene, viewing geometry) enters here", fontsize=6.5, color=INK2)
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------- Fig 2: analytical collapse
def fig_collapse():
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "sim", "swarm", "ispat"))
    from d1_revizyon_sim import p_analitik
    N, pfp = 8, 0.10; rr = np.linspace(0, 0.95, 60)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
    for Q, c, mk in ((2, C1, "o"), (4, C2, "s"), (6, C3, "D"), (8, C5, "^")):
        ax[0].plot(rr, [p_analitik(N, pfp, r, Q) for r in rr], color=c, marker=mk, ms=3, markevery=8, label=f"Q = {Q}")
    ax[0].axhline(0.01, color=INK, ls="--", lw=1); ax[0].text(0.55, 0.013, "1% target", fontsize=6.5)
    ax[0].set(yscale="log", xlabel="inter-agent correlation ρ", ylabel="P(false commit)", title="(a) False commit, N = 8, p_fp = 0.10")
    ax[0].legend(fontsize=6.5, loc="lower right")
    req = []
    for r in rr:
        q = [Q for Q in range(1, N + 1) if p_analitik(N, pfp, r, Q) <= 0.01]
        req.append(q[0] if q else np.nan)
    ax[1].step(rr, req, where="post", color=C1)
    rc = S["S1"]["kritik_rho_oybirligi"]
    ax[1].axvspan(rc, 0.95, color=C2, alpha=.12, lw=0); ax[1].text(rc + .02, 2, f"no threshold\nreaches 1%\n(ρ > {rc:.2f})", fontsize=6.5, color=INK)
    ax[1].set(xlabel="inter-agent correlation ρ", ylabel="smallest Q reaching 1%", title="(b) Required quorum", ylim=(0, 8.5))
    save(fig, "fig2_collapse.png")


if __name__ == "__main__":
    fig_arch(); fig_collapse()
    fig_same(); fig_cross(); fig_robust(); fig_rules(); fig_div()
