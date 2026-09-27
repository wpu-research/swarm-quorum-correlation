#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revizyon simülasyonları — sonuclar_rev/revizyon.json'daki ÖLÇÜLEN çekirdeklerle.

    python3 d1_revizyon_sim_kos.py        # → sonuclar_rev/simulasyon.json

S1  analitik (Eş. 2): gereken Q, oybirliğinin %1'i kaçırdığı kritik ρ
S2  tek model, geometri taraması; çekirdek bileşen ayrıştırması
S3  model çeşitliliği K ∈ {1,2,4,8}; ölçülen K×K çekirdekle, tüm alt kümeler
S4  LR vs eşik ailesi — ikisi de tam bütçede (ROC alt örtüsü)
S5  2×2: {tek küme, eşit} × {K=1, K=8}
"""
from __future__ import annotations
import json, os, sys, time
from itertools import combinations
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_revizyon_sim import (geometri, R_olustur, kararlar, say_kurali, lr_kurali, sabit_Q, n_eff_kurali,
                             p_analitik, kritik_rho)

SONUC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonuclar_rev")
TOHUM = 20260925
N = 8
P_YP, P_DOG = 0.10, 0.85
BUTCE = 0.20
HEDEF = 0.01
MC = int(os.environ.get("MC", 400_000))
EGITIM = int(os.environ.get("EGITIM", 2_000_000))
GEOM = ("esit", "yarim", "iki_kume", "tek_kume")


def yukle(sinif="tuzak", tip=None):
    """Çekirdekler: ÖLÇÜLEN ampirik eğriler (parametrik fit bazı modellerde tanımsız
    kalıyor; betimsel özet olarak raporlanır, simülasyona girmez)."""
    from d1_revizyon_sim import ampirik
    tip = tip or os.environ.get("CEKIRDEK", "ampirik")
    with open(os.path.join(SONUC, os.environ.get("REV", "revizyon.json"))) as f:
        R = json.load(f)
    ay = R["ayni_model"][sinif]["kosul"]
    cx = R["capraz"]["kosul"] if sinif == "tuzak" else R["capraz_sinif"][sinif]
    def k(e, ayni):
        return ampirik(e["egri"], ayni) if tip == "ampirik" else tuple(e["fit"])
    K_yp = {}
    for m, e in ay.items():
        try: K_yp[(m, m)] = k(e, True)
        except ValueError: pass
    for kk, e in cx.items():
        a, b = kk.split("|")
        try: K_yp[(a, b)] = k(e, False)
        except ValueError: pass
    K_dog = {}
    for m, e in R["hedef"]["ayni"].items():
        try: K_dog[(m, m)] = k(e, True)
        except ValueError: pass
    for kk, e in R["hedef"]["capraz"].items():
        a, b = kk.split("|")
        try: K_dog[(a, b)] = k(e, False)
        except ValueError: pass
    # hedef çekirdeği ölçülemeyen çiftler (p_dog≈1 → bilgisiz): ölçülebilen çiftlerin
    # medyan eğrisi (aynı-model / çapraz ayrı) — makalede beyan edilir
    if tip == "ampirik":
        from d1_revizyon_sim import ampirik
        def med(kaynak, ayni):
            E = [np.array([np.nan if v is None else v for v in e["egri"]], float) for e in kaynak]
            E = [x for x in E if (~np.isnan(x)).sum() >= 3]
            return ampirik(list(np.nanmedian(np.array(E), 0)), ayni)
        K_dog["_ayni"] = med(R["hedef"]["ayni"].values(), True)
        K_dog["_capraz"] = med(R["hedef"]["capraz"].values(), False)
    return R, K_yp, K_dog


class Varsayilanli(dict):
    """(a,b) yoksa (b,a), o da yoksa aynı/çapraz medyan hedef çekirdeği."""
    def get(self, k, d=None):
        if dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        return d
    def __getitem__(self, k):
        if dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        a, b = k
        if dict.__contains__(self, (b, a)):
            return dict.__getitem__(self, (b, a))
        return dict.__getitem__(self, "_ayni" if a == b else "_capraz")


def calis(b, mods, K_yp, K_dog, rng, p_yp=None, p_dog=None, lr=False, hedef_cekirdegi="olculen"):
    p_yp = np.full(len(b), P_YP) if p_yp is None else np.asarray(p_yp)
    p_dog = np.full(len(b), P_DOG) if p_dog is None else np.asarray(p_dog)
    Ry = R_olustur(b, mods, K_yp)
    Rd = R_olustur(b, mods, Varsayilanli(K_dog) if hedef_cekirdegi == "olculen" else K_yp)
    ky = kararlar(Ry, p_yp, MC, rng); kd = kararlar(Rd, p_dog, MC, rng)
    out = dict(esik_butce=say_kurali(ky, kd, BUTCE))
    for Q in (4, 5, 6):
        fa, ms = sabit_Q(ky, kd, Q); out[f"Q{Q}"] = (fa, ms)
    if lr:
        out["lr_butce"] = lr_kurali(Ry, Rd, p_yp, p_dog, ky, kd, BUTCE, EGITIM, rng)
        th = 30.0                                     # n_eff ölçeği: ölçülen θ_c mertebesi
        out["neff_butce"] = n_eff_kurali(b, th, ky, kd, BUTCE)
    return out


def main():
    S = {"N": N, "p_yp": P_YP, "p_dog": P_DOG, "butce": BUTCE, "MC": MC}
    for sinif in ("tuzak", "kus", "direk"):
        S[sinif] = calis_sinif(sinif)
    S.update(S["tuzak"])                                  # geriye uyum: üst düzey = tüm tuzaklar
    with open(os.path.join(SONUC, os.environ.get("SIM", "simulasyon.json")), "w") as f:
        json.dump(S, f, indent=1, default=float)
    print("→ simulasyon.json")


def calis_sinif(sinif):
    t0 = time.time()
    R, K_yp, K_dog = yukle(sinif)
    rng = np.random.default_rng(TOHUM)
    S = {}
    print(f"== sınıf {sinif}")

    # ---------------- S1 analitik
    S["S1"] = dict(
        Q_bagimsiz=[Q for Q in range(1, N + 1) if p_analitik(N, P_YP, 0, Q) <= HEDEF][0],
        oybirligi_rho08=p_analitik(N, P_YP, 0.8, N),
        kritik_rho_oybirligi=kritik_rho(N, P_YP, HEDEF),
        tablo=[dict(rho=r, **{f"Q{Q}": p_analitik(N, P_YP, r, Q) for Q in range(1, N + 1)})
               for r in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9)])

    # ---------------- S2 tek model, geometri; bileşen ayrıştırması (model A)
    S["S2"] = {}
    _, K_fit, _ = yukle(sinif, "fit")
    for m in [k[0] for k in K_yp if k[0] == k[1]]:
        w, v, th, mi = K_fit[(m, m)]
        varyant = {"olculen": (w, v, th, mi), "yalniz_bakis": (0.0, v, th, mi),
                   "yalniz_kosul": (w, 0.0, th, 0.0), "aynasiz": (w, v, th, 0.0)}
        for vad, par in varyant.items():
            if m not in ("A", "yolo11n") and vad != "olculen":   # bileşen ayrıştırması yalnız tek model
                continue
            Kv = dict(K_yp); Kv[(m, m)] = par if vad != "olculen" else K_yp[(m, m)]
            for g in GEOM:
                b = geometri(g, N)
                S["S2"].setdefault(m, {}).setdefault(vad, {})[g] = calis(b, [m] * N, Kv, K_dog, rng,
                                                                        lr=(vad == "olculen" and m == "A"))
    print(f"S2 bitti ({time.time()-t0:.0f} s)")

    # ---------------- S3 model çeşitliliği
    aday = sorted({k[0] for k in K_yp if k[0] == k[1]} - {"A"} - {m for m in {k[0] for k in K_yp} if m.endswith("@640")})
    havuz = [m for m in aday if all((m, o) in K_yp or (o, m) in K_yp for o in aday if o != m)]
    S["S3_havuz_disi"] = sorted(set(aday) - set(havuz))
    # A ≡ yolo11n (aynı ağırlık); @640 kontrolleri ayrı raporlanır
    S["S3"] = {"havuz": havuz}
    for g in ("esit", "tek_kume"):
        b = geometri(g, N)
        for K in ((1, 2, 4, 8) if len(havuz) >= 8 else (1, 2, 4, len(havuz))):
            if K > len(havuz):
                continue
            alt = list(combinations(havuz, K))
            if len(alt) > 40:
                alt = [alt[i] for i in np.random.default_rng(K).choice(len(alt), 40, replace=False)]
            sonuc = []
            for s in alt:
                mods = [s[i % K] for i in range(N)]
                r = calis(b, mods, K_yp, K_dog, rng)
                sonuc.append(dict(modeller=list(s), **r))
            S["S3"].setdefault(g, {})[K] = sonuc
        print(f"S3 {g} bitti ({time.time()-t0:.0f} s)")

    # ---------------- S4 LR vs eşik: tek model ve K=8, geometriler
    S["S4"] = {}
    k8 = havuz[:8] if len(havuz) >= 8 else havuz
    for g in GEOM:
        b = geometri(g, N)
        tek = "A" if ("A", "A") in K_yp else "yolo11n"
        for ad, mods in (("K1_A", [tek] * N), ("K8", [k8[i % len(k8)] for i in range(N)])):
            S["S4"].setdefault(g, {})[ad] = calis(b, mods, K_yp, K_dog, rng, lr=True)
    print(f"S4 bitti ({time.time()-t0:.0f} s)")

    return S


if __name__ == "__main__":
    main()
