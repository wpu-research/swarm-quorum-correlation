#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revizyon analizinin tamamı — tek komut, tüm sayılar ve şekiller.

    python3 d1_revizyon_kos.py            # → sonuclar_rev/revizyon.json + *.csv + sekil_*.png

Kaynaklar (hepsi aynı tohumlu kareler; kayit_coklu yoksa yalnız A/B ile koşar):
  A      kayit/tam     yolo11n (ultralytics, x86)          tespit_var @0.40
  B      kayit/hailo   yolov8s_h8.hef (Hailo-8, RPi5)      tespit_var @0.40
  <m>    kayit_coklu/ortak/*.npz      9 COCO dedektörü, güven → eşik
  <m>'   kayit_coklu/bagimsiz/*.npz   aynı modeller, bozulma açıya göre bağımsız
  drone  kayit_drone/{ortak,bagimsiz} drone-eğitimli yolo11n (Bird/Drone)
"""
from __future__ import annotations
import json, os, sys, time, warnings
from itertools import combinations
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_revizyon import (KOK, KUTULAR, meta, csv_kaynak, npz_guven, esikle, sayimlar,
                         egri_sayimdan, bootstrap_agirliklari, oturt, orta_bant, cekirdek)

warnings.filterwarnings("ignore")
SONUC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonuclar_rev")
os.makedirs(SONUC, exist_ok=True)
B_BOOT = int(os.environ.get("B_BOOT", 2000))
RNG = np.random.default_rng(20260925)
CIFT_KARE = np.arange(0, 60, 2)                # yeni koşular çift kareleri işler


def cift_karelere_indir(src):
    """60 kareli kaynakları (A, B, eski npz) 30 çift kareye indir → tüm kaynaklar
    aynı kare kümesinde karşılaştırılır."""
    out = {}
    for d, (a, T) in src.items():
        out[d] = (a, T[:, CIFT_KARE] if T.shape[1] == 60 else T)
    return out


MOD = os.environ.get("MOD", "ortak")          # "bagimsiz": yalnız bağımsız-bozulma kaynakları, ekler olmadan


def yukle():
    if MOD == "bagimsiz":
        guven = {}
        for kok in ("kayit_coklu2/bagimsiz", "kayit_coklu3/bagimsiz", "kayit_drone/bagimsiz", "kayit_640b/bagimsiz"):
            for m, g in npz_guven(kok).items():
                ad = "drone" if m.startswith("drone") else m
                if kok.startswith("kayit_640b") and not ad.startswith("rtdetr"):
                    ad += "@640"
                guven[ad] = {d: (a, c[:, CIFT_KARE] if c.shape[1] == 60 else c) for d, (a, c) in g.items()}
        return {}, guven
    kay = {}
    kay["A"] = cift_karelere_indir(csv_kaynak("kayit/tam"))
    kay["B"] = cift_karelere_indir(csv_kaynak("kayit/hailo"))
    guven = {}
    for kok, ek in (("kayit_coklu/ortak", ""), ("kayit_coklu2/bagimsiz", "'"), ("kayit_drone/ortak", ""),
                    ("kayit_drone/bagimsiz", "'"), ("kayit_rtdetr/ortak", ""), ("kayit_640/ortak", "@640")):
        for m, g in npz_guven(kok).items():
            ad = ("drone" if m.startswith("drone") else m)
            ad = ad + ek if ek != "@640" else ad + "@640"
            guven[ad] = {d: (a, c[:, CIFT_KARE] if c.shape[1] == 60 else c) for d, (a, c) in g.items()}
    return kay, guven


def ortak_denemeler(kaynaklar, siniflar):
    ids = None
    for s in kaynaklar.values():
        k = {d for d in s if meta(d) and meta(d)["sinif"] in siniflar}
        ids = k if ids is None else ids & k
    return sorted(ids or [])


# ------------------------------------------------------------------ özetler
def marjinaller(kaynaklar):
    out = {}
    for m, s in kaynaklar.items():
        t = {}
        for d, (a, T) in s.items():
            mm = meta(d); t.setdefault(mm["sinif"], {}).setdefault(mm["mes"], []).append(T.mean())
        out[m] = {c: {str(r): round(float(np.mean(v)), 4) for r, v in sorted(rr.items())} for c, rr in t.items()}
        out[m]["_tuzak_ort"] = round(float(np.mean([T.mean() for d, (a, T) in s.items()
                                                    if meta(d)["sinif"] in ("kus", "insan", "direk")])), 4)
    return out


def egri_ve_ga(C, ids, havuz, kume="katman", B=B_BOOT):
    r0, _, _ = egri_sayimdan(C, ids, havuz)
    W = bootstrap_agirliklari(ids, B, RNG, kume)
    rb = np.stack([egri_sayimdan(C, ids, havuz, W[b])[0] for b in range(B)])
    return r0, rb


def fit_ga(r0, rb, ayna=True, sifir_dahil=False):
    p0 = oturt(r0, sifir_dahil=sifir_dahil, ayna=ayna)
    pb = np.array([oturt(r, sifir_dahil=sifir_dahil, ayna=ayna) for r in rb])
    pb = pb[~np.isnan(pb).any(1)] if pb.ndim == 2 else np.empty((0, 4))
    if len(pb) == 0:                                   # bilgisiz eğri (ör. p_dog ≈ 1)
        return p0, np.full((2, 4), np.nan), pb
    return p0, np.percentile(pb, [2.5, 97.5], axis=0), pb


def yuvarla(x, k=4):
    if isinstance(x, dict):
        return {kk: yuvarla(v, k) for kk, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [yuvarla(v, k) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), k)
    return x


# ------------------------------------------------------------------ ana
def main():
    t0 = time.time()
    kay, guven = yukle()
    # eşik 0.40 (orijinal protokol) — npz kaynakları ikili tespite
    for m, g in guven.items():
        kay[m] = esikle(g, 0.40)
    ref = "A" if "A" in kay else "yolo11n"
    tam = set(ortak_denemeler({ref: kay[ref]}, ("kus", "insan", "direk", "drone")))
    for m in list(kay):
        eksik = tam - set(kay[m])
        if eksik:
            print(f"  UYARI: {m} eksik ({len(eksik)} deneme) — analizden çıkarıldı")
            del kay[m]
    R = {"kaynaklar": sorted(kay), "B_bootstrap": B_BOOT, "kareler": "çift kareler (30/klip)"}
    R["marjinaller"] = marjinaller(kay)
    if "yolo11n" in kay and "A" in kay:   # yeniden üretim kontrolü: A (x86, fp32) vs panel yolo11n (GPU, fp16)
        ag = [(kay["A"][d][1] == kay["yolo11n"][d][1]).mean() for d in kay["A"] if d in kay["yolo11n"] and meta(d)["sinif"] in ("kus", "insan", "direk", "drone")]
        R["uyum_A_yolo11n"] = dict(ort=float(np.mean(ag)), min=float(np.min(ag)), n=len(ag))
    print("kaynaklar:", sorted(kay), f"({time.time()-t0:.0f} s)")

    # ---------------- 1. aynı-model eğrileri: havuz düzeyi × sınıf (A, B ve npz)
    modeller = [m for m in kay if not m.endswith("'")]
    sinif_setleri = {"tuzak": ("kus", "insan", "direk"), "kus": ("kus",), "direk": ("direk",)}
    R["ayni_model"] = {}
    for sad, ss in sinif_setleri.items():
        ids = ortak_denemeler({m: kay[m] for m in modeller}, ss)
        C = sayimlar(kay, ids, [(m, m) for m in modeller])
        for havuz in ("deneme", "isik", "kosul"):
            for mi, m in enumerate(modeller):
                ana = m in ("A", "B", "yolo11n")
                Bk = B_BOOT if (havuz == "kosul" and ana) else (400 if havuz == "kosul" or ana else 200)
                r0, rb = egri_ve_ga(C[mi], ids, havuz, B=Bk)
                ob = orta_bant(r0); obb = orta_bant(rb)
                ent = dict(egri=r0, egri_ga=np.nanpercentile(rb, [2.5, 97.5], 0),
                           orta_bant=ob, orta_bant_ga=np.nanpercentile(obb, [2.5, 97.5]),
                           p_orta_bant_le0=float(np.mean(obb <= 0)))
                if havuz == "kosul":
                    p0, ga, _ = fit_ga(r0, rb)
                    ent.update(fit=p0, fit_ga=ga)
                    if sad == "tuzak":
                        p1, ga1, _ = fit_ga(r0, rb, ayna=False)
                        ent.update(fit_tek=p1, fit_tek_ga=ga1)
                R["ayni_model"].setdefault(sad, {}).setdefault(havuz, {})[m] = ent
        print(f"aynı-model {sad} bitti ({time.time()-t0:.0f} s)")

    # ---------------- 1b. hedef (drone) sınıfı: aynı ve çapraz model, kosul havuzu
    ids_h = ortak_denemeler({m: kay[m] for m in modeller}, ("drone",))
    Ch = sayimlar(kay, ids_h, [(m, m) for m in modeller])
    R["hedef"] = {"ayni": {}, "capraz": {}}
    for mi, m in enumerate(modeller):
        r0, rb = egri_ve_ga(Ch[mi], ids_h, "kosul", B=400)
        p0, ga, _ = fit_ga(r0, rb)
        R["hedef"]["ayni"][m] = dict(egri=r0, fit=p0, fit_ga=ga)
    Chx = sayimlar(kay, ids_h, list(combinations(modeller, 2)))
    for pi, (a, b) in enumerate(combinations(modeller, 2)):
        r0, rb = egri_ve_ga(Chx[pi], ids_h, "kosul", B=200)
        p0, ga, _ = fit_ga(r0, rb, sifir_dahil=True)
        R["hedef"]["capraz"][f"{a}|{b}"] = dict(egri=r0, fit=p0, fit_ga=ga)
    print(f"hedef bitti ({time.time()-t0:.0f} s)")

    # ---------------- 2. ayna-kopya çiftleri dışlanmış duyarlılık (tuzak, kosul)
    ids = ortak_denemeler({m: kay[m] for m in modeller}, sinif_setleri["tuzak"])
    C_ay = sayimlar(kay, ids, [(m, m) for m in modeller], ayna_disla=True)
    R["ayna_dislanmis"] = {}
    for mi, m in enumerate(modeller):
        r0, rb = egri_ve_ga(C_ay[mi], ids, "kosul", B=400)
        R["ayna_dislanmis"][m] = dict(egri=r0, orta_bant=orta_bant(r0),
                                      orta_bant_ga=np.nanpercentile(orta_bant(rb), [2.5, 97.5]))

    # ---------------- 3. çapraz-model matrisi (tuzak, kosul ve deneme havuzu)
    ciftler = list(combinations(modeller, 2))
    C_x = sayimlar(kay, ids, ciftler)
    R["capraz"] = {}
    for havuz in ("deneme", "kosul"):
        for pi, (a, b) in enumerate(ciftler):
            ana = {a, b} in ({"A", "B"}, {"A", "yolo11n@640"}, {"B", "yolov8s@640"}, {"yolov8s", "yolov8s@640"})
            B = B_BOOT if ana else (300 if havuz == "kosul" else 100)
            r0, rb = egri_ve_ga(C_x[pi], ids, havuz, B=B)
            ent = dict(egri=r0, rho0=r0[0], rho0_ga=np.nanpercentile(rb[:, 0], [2.5, 97.5]),
                       orta_bant=orta_bant(r0), orta_bant_ga=np.nanpercentile(orta_bant(rb), [2.5, 97.5]))
            if havuz == "kosul":
                p0, ga, _ = fit_ga(r0, rb, sifir_dahil=True)
                ent.update(fit=p0, fit_ga=ga)
            R["capraz"].setdefault(havuz, {})[f"{a}|{b}"] = ent
    print(f"çapraz bitti ({time.time()-t0:.0f} s)")

    # sınıf bazında çapraz çekirdekler (simülasyon için; kosul havuzu)
    R["capraz_sinif"] = {}
    for sad in ("kus", "direk"):
        idc = ortak_denemeler({m: kay[m] for m in modeller}, sinif_setleri[sad])
        Cc = sayimlar(kay, idc, ciftler)
        for pi, (a, b) in enumerate(ciftler):
            r0, rb = egri_ve_ga(Cc[pi], idc, "kosul", B=100)
            p0, ga, _ = fit_ga(r0, rb, sifir_dahil=True)
            R["capraz_sinif"].setdefault(sad, {})[f"{a}|{b}"] = dict(egri=r0, fit=p0, orta_bant=orta_bant(r0))
    print(f"çapraz sınıf bitti ({time.time()-t0:.0f} s)")

    # aynı ile çapraz farkı — A/B için eşleşmiş bootstrap
    R["fark_AB"] = None
    if "A" in kay:
        Cab = np.stack([sayimlar(kay, ids, [("A", "A")])[0], sayimlar(kay, ids, [("B", "B")])[0],
                        sayimlar(kay, ids, [("A", "B")])[0]])
        W = bootstrap_agirliklari(ids, B_BOOT, RNG, "katman")
        fark = []
        for b in range(B_BOOT):
            r = egri_sayimdan(Cab, ids, "kosul", W[b])[0]
            ob = orta_bant(r); fark.append([ob[0] - ob[2], ob[1] - ob[2]])
        fark = np.array(fark)
        R["fark_AB"] = dict(AA_eksi_AB_ga=np.percentile(fark[:, 0], [2.5, 50, 97.5]),
                            BB_eksi_AB_ga=np.percentile(fark[:, 1], [2.5, 50, 97.5]),
                            p_AA_le_AB=float(np.mean(fark[:, 0] <= 0)), p_BB_le_AB=float(np.mean(fark[:, 1] <= 0)))

    # ---------------- 4. bozulma ablation'ı: ortak vs bağımsız bozulma
    ab = [m for m in kay if m.endswith("'")]
    R["ablation"] = {}
    for mp in ab:
        m = mp[:-1]
        if m not in kay:
            continue
        idsab = ortak_denemeler({m: kay[m], mp: kay[mp]}, sinif_setleri["tuzak"])
        Cm = sayimlar(kay, idsab, [(m, m), (mp, mp)])
        for havuz in ("deneme", "kosul"):
            r_o, rb_o = egri_ve_ga(Cm[0], idsab, havuz, B=400)
            r_b, rb_b = egri_ve_ga(Cm[1], idsab, havuz, B=400)
            R["ablation"].setdefault(havuz, {})[m] = dict(
                ortak=r_o, bagimsiz=r_b, ortak_orta=orta_bant(r_o), bagimsiz_orta=orta_bant(r_b),
                ortak_orta_ga=np.nanpercentile(orta_bant(rb_o), [2.5, 97.5]),
                bagimsiz_orta_ga=np.nanpercentile(orta_bant(rb_b), [2.5, 97.5]),
                n_deneme=len(idsab))
    print(f"ablation bitti ({time.time()-t0:.0f} s)")

    # ---------------- 5. kopula geçerliliği: tetrakorik eşikten bağımsız mı?
    R["esik_degismezligi"] = {}
    for m in (("yolo11n", "yolov8s", "rtdetr-l") if MOD == "ortak" else ()):   # drone: marjinal ~%1, tetrakorik tanımsız
        if m not in guven:
            continue
        ks = {f"{m}@{e}": esikle(guven[m], e) for e in (0.25, 0.40, 0.55)}
        idsE = ortak_denemeler(ks, sinif_setleri["tuzak"])
        CE = sayimlar(ks, idsE, [(k, k) for k in ks])
        for ki, k in enumerate(ks):
            r0, rb = egri_ve_ga(CE[ki], idsE, "kosul", B=400)
            R["esik_degismezligi"][k] = dict(egri=r0, orta_bant=orta_bant(r0),
                                             orta_bant_ga=np.nanpercentile(orta_bant(rb), [2.5, 97.5]),
                                             p_ort=float(np.mean([T.mean() for d, (a, T) in ks[k].items() if d in idsE])))

    # ---------------- 6. nesne düzeyinde küme bootstrap'ı (duyarlılık; az küme → aralık liberal)
    if "A" in kay:
        idsA = ortak_denemeler({"A": kay["A"]}, sinif_setleri["tuzak"])
        CA = sayimlar(kay, idsA, [("A", "A")])[0]
        r0, rb = egri_ve_ga(CA, idsA, "kosul", kume="ornek", B=B_BOOT)
        ob = orta_bant(rb)
        R["ornek_bootstrap_A"] = dict(orta_bant=float(orta_bant(r0)), ga=np.nanpercentile(ob, [2.5, 97.5]),
                                      p_le0=float(np.mean(ob <= 0)), n_ornek=len({meta(d)["ornek"] for d in idsA}))

    with open(os.path.join(SONUC, "revizyon.json" if MOD == "ortak" else f"revizyon_{MOD}.json"), "w") as f:
        json.dump(yuvarla(R), f, indent=1, ensure_ascii=False)
    print(f"→ {SONUC}/revizyon.json  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
