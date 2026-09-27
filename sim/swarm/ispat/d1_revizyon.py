#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1 revizyon analizi (hakem turu, 2026-09-25) — tek kestirici, tüm karşılaştırmalar.

Eski analizdeki sorunlar ve buradaki karşılıkları:
  1. Aynı-model eğrisi katman (örnek×mesafe; fon+ışık HAVUZLU) içinde, çapraz-model
     eğrisi DENEME içinde kestiriliyordu → iki büyüklük aynı ölçüt değildi.
     Burada ikisi de AYNI kestiriciyle ve açıkça seçilen havuzlama düzeyinde.
  2. ρ_ağırlık alttan 0'a sınırlıydı → sınırda bootstrap. Burada sınırsız fit,
     B=2000 bootstrap, H0: taban=0.
  3. Δθ ≤ 90° kesmesi 180°'deki ayna yükselişini gizliyordu → tüm 0–180°,
     ayna terimli çekirdek; birebir ayna-kopya görünüş çiftleri ayrıca dışlanabilir.
  4. Sınıf karışımı → sınıf bazında eğri.

Kestirici (sayım tabanlı, vektörize):
  deneme × Δθ kutusu × model çifti için (n, ΣX, ΣY, ΣXY) sayımları tutulur.
  Havuz = bir gruptaki denemelerin sayımlarının toplamı → tetrakorik.
  Katman (örnek×mesafe) ortalaması → eğri. Bootstrap = deneme ağırlıklarıyla
  aynı toplam (yeniden sayım yok) → binlerce tekrar saniyeler içinde.
"""
from __future__ import annotations
import csv, glob, os, re
from collections import defaultdict
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TUZAK_ACILI = ("kus", "insan", "direk")
KUTULAR = np.arange(0, 181, 15.0)


# ------------------------------------------------------------------ meta
def meta(did):
    m = re.match(r"^(?P<ornek>.+?)_(?P<mes>\d+)m_(?P<arka>gok|kent)_(?P<isik>parlak|kapali)$", did)
    if not m:
        return None
    o = m["ornek"]
    return dict(ornek=o, sinif=o.split("_")[0], mes=int(m["mes"]), arka=m["arka"], isik=m["isik"])


HAVUZ = {   # grup anahtarı: aynı gruptaki denemeler tek tetrakoriğe havuzlanır
    "deneme": lambda m: (m["ornek"], m["mes"], m["arka"], m["isik"]),
    "isik":   lambda m: (m["ornek"], m["mes"], m["arka"]),          # ışık ortak faktör
    "fon":    lambda m: (m["ornek"], m["mes"], m["isik"]),
    "kosul":  lambda m: (m["ornek"], m["mes"]),                     # ışık+fon (birincil)
}


# ------------------------------------------------------------------ yükleme
def csv_kaynak(kok):
    """kayit/<oturum>/*/kareler.csv → {deneme: (açılar, tespit[V,F] bool)}"""
    out = {}
    for yol in sorted(glob.glob(os.path.join(KOK, kok, "*", "kareler.csv"))):
        did = os.path.basename(os.path.dirname(yol))
        if meta(did) is None:
            continue
        a = defaultdict(dict)
        with open(yol) as f:
            for s in csv.DictReader(f):
                a[float(s["aci_derece"])][int(s["kare_no"])] = int(s["tespit_var"])
        A = np.array(sorted(a)); F = sorted(a[A[0]])
        out[did] = (A, np.array([[a[x][k] for k in F] for x in A], bool))
    return out


def npz_guven(kok):
    """kayit_coklu/<mod>/*.npz → {model: {deneme: (açılar, güven[V,F] float32)}}"""
    out = defaultdict(dict)
    for yol in sorted(glob.glob(os.path.join(KOK, kok, "*.npz"))):
        did = os.path.basename(yol)[:-4]
        if meta(did) is None:
            continue
        z = np.load(yol); o = np.argsort(z["aci"])
        for i, m in enumerate(z["modeller"]):
            out[str(m).replace(".pt", "")][did] = (z["aci"][o], z["conf"][i][o].astype(np.float32))
    return dict(out)


def esikle(guven, esik):
    return {d: (a, c >= esik) for d, (a, c) in guven.items()}


# ------------------------------------------------------------------ vektörize tetrakorik
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def bvn_ust(h, k, r):
    """P(Z1>h, Z2>k; r) — Plackett integrali, Gauss–Legendre (vektörize)."""
    h, k, r = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float), np.asarray(r, float))
    t = (_GL_X[:, None] + 1) / 2 * r.ravel()[None, :]                      # (32, n) 0..r
    hh, kk = h.ravel()[None, :], k.ravel()[None, :]
    g = np.exp(-(hh * hh - 2 * hh * kk * t + kk * kk) / (2 * (1 - t * t))) / np.sqrt(1 - t * t)
    integ = (g * _GL_W[:, None]).sum(0) * r.ravel() / 2 / (2 * np.pi)
    return (norm.sf(h.ravel()) * norm.sf(k.ravel()) + integ).reshape(h.shape)


def tetrakorik_v(p1, p2, p11, iter_=40):
    """Vektörize ikiye bölme: Φ2(τ1,τ2;ρ)=p11."""
    p1, p2, p11 = (np.asarray(x, float) for x in (p1, p2, p11))
    t1, t2 = norm.isf(np.clip(p1, 1e-6, 1 - 1e-6)), norm.isf(np.clip(p2, 1e-6, 1 - 1e-6))
    lo = np.full(p1.shape, -0.995); hi = np.full(p1.shape, 0.995)
    for _ in range(iter_):
        mid = (lo + hi) / 2
        f = bvn_ust(t1, t2, mid) - p11
        hi = np.where(f > 0, mid, hi); lo = np.where(f > 0, lo, mid)
    return (lo + hi) / 2


# ------------------------------------------------------------------ açı çiftleri
AYNA = {   # birebir ayna-kopya / tekrar görünüşler (görüntü benzerliği > 0.97)
    "kus_karga": {(0, 180), (195, 345), (210, 330), (225, 315)},
    "kus_guvercin": {(0, 180), (15, 165), (30, 150), (45, 135), (60, 120), (195, 345), (210, 330), (225, 315)},
    "insan_01": {(0, 345), (15, 135), (15, 165), (30, 60), (30, 120), (60, 120), (90, 105), (135, 165),
                 (210, 315), (210, 330), (315, 330)},
    "insan_02": {(30, 150), (60, 120)},
    "insan_03": {(15, 165), (30, 45), (30, 135), (30, 180), (45, 135), (210, 330)},
    "drone_01": {(45, 135), (210, 330), (225, 315)},
}


def cift_maskesi(acilar, c, ayni_model, ornek=None, ayna_disla=False):
    d = np.abs(acilar[:, None] - acilar[None, :]) % 360
    d = np.minimum(d, 360 - d)
    if c == 0:
        if ayni_model:
            return None
        sel = np.eye(len(acilar), dtype=bool)
    else:
        sel = np.abs(d - c) < 7.5
        if ayni_model:
            sel &= np.triu(np.ones_like(sel), 1).astype(bool)
    if ayna_disla and ornek in AYNA:
        for a, b in AYNA[ornek]:
            ia = np.where(acilar == a)[0]; ib = np.where(acilar == b)[0]
            if ia.size and ib.size:
                sel[ia[0], ib[0]] = sel[ib[0], ia[0]] = False
    return sel if sel.any() else None


# ------------------------------------------------------------------ sayımlar
def sayimlar(kaynaklar, ids, ciftler, ayna_disla=False):
    """kaynaklar: {model: {deneme: (açı, tespit[V,F])}}; ciftler: [(a, b)].
    Döner C[çift, deneme, kutu, 4] = (n, ΣX, ΣY, ΣXY)."""
    C = np.zeros((len(ciftler), len(ids), len(KUTULAR), 4))
    maske = {}
    for pi, (ma, mb) in enumerate(ciftler):
        ayni = ma == mb
        for ti, did in enumerate(ids):
            acA, TA = kaynaklar[ma][did]; acB, TB = kaynaklar[mb][did]
            assert np.array_equal(acA, acB), did
            orn = meta(did)["ornek"]
            TA = TA.astype(np.float64); TB = TB.astype(np.float64)
            for ci, c in enumerate(KUTULAR):
                k = (orn, len(acA), ci, ayni)
                if k not in maske:
                    maske[k] = cift_maskesi(acA, c, ayni, orn, ayna_disla)
                sel = maske[k]
                if sel is None:
                    continue
                I, J = np.where(sel)
                X = TA[I]; Y = TB[J]
                C[pi, ti, ci] = (X.size, X.sum(), Y.sum(), (X * Y).sum())
    return C


def egri_sayimdan(C, ids, havuz="kosul", agirlik=None, pmin=0.02, min_katman=2):
    """C[..., deneme, kutu, 4] → eğri[..., kutu]. agirlik[deneme] bootstrap için.
    Havuz grubu içinde topla → tetrakorik → katman (örnek×mesafe) içinde grup
    ortalaması → katmanlar arası eşit ağırlıklı ortalama."""
    ms = [meta(d) for d in ids]
    grup = defaultdict(list)
    for ti, m in enumerate(ms):
        grup[HAVUZ[havuz](m)].append(ti)
    gruplar = list(grup)
    katman_of = [(ms[grup[g][0]]["ornek"], ms[grup[g][0]]["mes"]) for g in gruplar]
    w = np.ones(len(ids)) if agirlik is None else agirlik
    S = np.stack([np.tensordot(w[grup[g]], np.take(C, grup[g], axis=-3), axes=([0], [-3])) for g in gruplar], -3)
    n = S[..., 0]; nn = np.maximum(n, 1)
    px, py, pxy = S[..., 1] / nn, S[..., 2] / nn, S[..., 3] / nn
    ok = (n > 0) & (px >= pmin) & (px <= 1 - pmin) & (py >= pmin) & (py <= 1 - pmin)
    rho = np.where(ok, tetrakorik_v(px, py, pxy), np.nan)                 # [..., grup, kutu]
    katmanlar = sorted(set(katman_of))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K = np.stack([np.nanmean(rho[..., [i for i, k in enumerate(katman_of) if k == kk], :], axis=-2)
                          for kk in katmanlar], -2)
            r = np.nanmean(K, axis=-2)
            # tek bir bilgili katmana dayanan kutu güvenilmez (uç marjinalde tetrakorik ±0.9'a kaçabiliyor)
            r = np.where(np.sum(~np.isnan(K), axis=-2) >= min_katman, r, np.nan)
            return r, K, katmanlar


def bootstrap_agirliklari(ids, B, rng, kume="katman"):
    """kume:
      "katman_ici": katman (örnek×mesafe) İÇİNDE deneme yeniden örneklemesi — orijinal
                    tasarım. 4 koşuldan geri koyarak çekmek koşullar arası varyansı
                    ~3/4'e indirir → koşul bileşeni sistematik olarak EKSİK kestirilir
                    (eski z0=1.83 sapmasının kaynağı). Yalnız karşılaştırma için.
      "katman":     katmanları BÜTÜN olarak yeniden örnekle (birincil; katman = ortalamanın birimi).
      "ornek":      nesne örneği düzeyinde küme bootstrap'ı (en tutucu; az küme)."""
    ms = [meta(d) for d in ids]
    if kume == "katman_ici":
        anahtar = lambda m: (m["ornek"], m["mes"])
    elif kume == "katman":
        anahtar = lambda m: (m["ornek"], m["mes"])
    else:
        anahtar = lambda m: m["ornek"]
    grup = defaultdict(list)
    for i, m in enumerate(ms):
        grup[anahtar(m)].append(i)
    W = np.zeros((B, len(ids)))
    if kume == "katman_ici":
        for g in grup.values():
            g = np.array(g); sec = rng.integers(0, len(g), (B, len(g)))
            for b in range(B):
                np.add.at(W[b], g[sec[b]], 1)
    else:
        ks = list(grup)
        for b in range(B):
            for g in rng.integers(0, len(ks), len(ks)):
                W[b, grup[ks[g]]] += 1
    return W


# ------------------------------------------------------------------ çekirdek
def cekirdek(d, w, b, th, m):
    """ρ(Δθ) = ρ_c + ρ_v·[exp(−Δθ/θ_c) + m·exp(−(180−Δθ)/θ_c)]
    ρ_c: bakıştan bağımsız bileşen (sınırsız — negatif olabilir)
    m:   ayna (bilateral simetri) ağırlığı; m=0 eski tek-yönlü çekirdek."""
    return w + b * (np.exp(-d / th) + m * np.exp(-(180 - d) / th))


def oturt(r, sifir_dahil=False, ayna=True):
    x = KUTULAR.copy()
    sel = ~np.isnan(r) & ((x > 0) | sifir_dahil)
    if sel.sum() < 5:
        return np.full(4, np.nan)
    if ayna:
        f = lambda p: cekirdek(x[sel], *p) - r[sel]
        s = least_squares(f, (0.05, 0.5, 30, 0.5), bounds=((-0.5, 0, 1, 0), (1, 2, 120, 1.5)))
        return s.x
    f = lambda p: cekirdek(x[sel], p[0], p[1], p[2], 0) - r[sel]
    s = least_squares(f, (0.05, 0.5, 30), bounds=((-0.5, 0, 1), (1, 2, 120)))
    return np.r_[s.x, 0.0]


def orta_bant(r, lo=60, hi=120):
    """Bakıştan bağımsız bileşenin çekirdekten bağımsız özeti: 60–120° ortalaması."""
    sel = (KUTULAR >= lo) & (KUTULAR <= hi)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(r[..., sel], axis=-1)
