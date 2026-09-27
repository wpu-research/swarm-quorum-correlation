#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Revizyon simülasyonu — ÖLÇÜLEN çekirdekler ve marjinallerle sürü karar kuralları.

Eski simülasyondan (makale_ispat.py D3/D7/D8) farkları:
  * Her ajanın kendi modeli, kendi marjinali (p_yp, p_dog) ve model-çifti başına
    ölçülen çekirdek: R_ij = K_{m_i m_j}(Δθ_ij), ayna terimi dahil, 0–180° tam.
  * Kurallar tam kaçırma bütçesinde karşılaştırılır: ayrık eşik ailesi komşu iki
    Q arasında randomize edilir (ROC dışbükey örtüsü), LR de aynı şekilde →
    "LR kazancı" ayrıklık artefaktı içermez.
  * LR desen olasılıkları bağımsız eğitim örnekleminden (train/test ayrı).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, binom

try:
    from d1_revizyon import cekirdek
except ImportError:                      # pragma: no cover
    def cekirdek(d, w, b, th, m):
        return w + b * (np.exp(-d / th) + m * np.exp(-(180 - d) / th))


# ------------------------------------------------------------------ geometri
def aci_farki(a, b):
    d = np.abs(np.asarray(a, float)[:, None] - np.asarray(b, float)[None, :]) % 360
    return np.minimum(d, 360 - d)


def geometri(ad, N, rng=None):
    rng = rng or np.random.default_rng(7)
    if ad == "esit":
        return np.linspace(0, 360, N, endpoint=False)
    if ad == "yarim":                   # tek yaklaşma yönü: 180° yay
        return np.linspace(0, 180, N)
    if ad == "iki_kume":
        return (np.repeat([0, 180], N // 2) + rng.normal(0, 5, N)) % 360
    if ad == "tek_kume":
        return (rng.normal(0, 10, N)) % 360
    raise KeyError(ad)


# ------------------------------------------------------------------ korelasyon
def psd_yap(R, eps=1e-6):
    R = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(R)
    if w.min() >= eps:
        return R
    R = V @ np.diag(np.clip(w, eps, None)) @ V.T
    d = np.sqrt(np.diag(R)); R = R / np.outer(d, d); np.fill_diagonal(R, 1.0)
    return R


def R_olustur(bearing, modeller, cekirdekler):
    """cekirdekler[(a, b)] = (w, v, θ, m) — a==b aynı-model, a!=b çapraz.
    Aynı model + Δθ→0: çekirdek ekstrapolasyonu, 1'e kırpılır."""
    D = aci_farki(bearing, bearing); n = len(bearing)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = modeller[i], modeller[j]
            p = cekirdekler.get((a, b)) or cekirdekler[(b, a)]   # Varsayilanli: medyan yedek
            v = p(D[i, j]) if callable(p) else cekirdek(D[i, j], *p)
            R[i, j] = R[j, i] = np.clip(v, -0.95, 0.95)
    return psd_yap(R)


def kararlar(R, p, boyut, rng):
    L = np.linalg.cholesky(R)
    Z = (L @ rng.standard_normal((R.shape[0], boyut))).T
    return Z > norm.isf(np.asarray(p, float))[None, :]


# ------------------------------------------------------------------ kurallar
def kod(k):
    return k.astype(np.int64) @ (1 << np.arange(k.shape[1])).astype(np.int64)


def roc_noktalari(skor_yp, skor_dog):
    """Skor ≥ t → commit. Tüm eşikler için (kaçırma, hatalı) noktaları (+ uçlar)."""
    ts = np.unique(np.concatenate([skor_yp, skor_dog]))
    sy = np.sort(skor_yp); sd = np.sort(skor_dog)
    fa = 1 - np.searchsorted(sy, ts, side="left") / sy.size
    ms = np.searchsorted(sd, ts, side="left") / sd.size
    return np.r_[ms, 1.0], np.r_[fa, 0.0]


def alt_ortu(x, y):
    """(x, y) noktalarının alt dışbükey örtüsü, x artan."""
    o = np.lexsort((y, x)); P = list(zip(x[o], y[o])); H = []
    for p in P:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (p[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (p[0] - H[-2][0]) <= 0:
            H.pop()
        H.append(p)
    return np.array([h[0] for h in H]), np.array([h[1] for h in H])


def butcede(skor_yp, skor_dog, butce):
    """Kaçırma ≤ butce koşulunda, randomize eşiklerle (ROC alt örtüsü) ulaşılabilen
    en düşük hatalı-commit olasılığı. Ayrık kural ile sürekli skor aynı zeminde."""
    ms, fa = roc_noktalari(skor_yp, skor_dog)
    hx, hy = alt_ortu(ms, fa)
    return float(np.interp(butce, hx, hy))


def say_kurali(kyp, kdg, butce):
    return butcede(kyp.sum(1).astype(float), kdg.sum(1).astype(float), butce)


def lr_kurali(R_yp, R_dog, p_yp, p_dog, kyp, kdg, butce, egitim, rng, duzeltme=0.5):
    n = kyp.shape[1]; M = 1 << n
    cy = np.bincount(kod(kararlar(R_yp, p_yp, egitim, rng)), minlength=M) + duzeltme
    cd = np.bincount(kod(kararlar(R_dog, p_dog, egitim, rng)), minlength=M) + duzeltme
    lam = np.log(cd / cd.sum()) - np.log(cy / cy.sum())
    return butcede(lam[kod(kyp)], lam[kod(kdg)], butce)


def sabit_Q(kyp, kdg, Q):
    return float((kyp.sum(1) >= Q).mean()), float((kdg.sum(1) < Q).mean())


# ------------------------------------------------------------------ analitik (Eş. 2)
_U = np.linspace(-9, 9, 40001); _PHI = norm.pdf(_U)


def p_analitik(n, p, rho, Q):
    if rho <= 0:
        return float(binom.sf(Q - 1, n, p))
    tau = norm.isf(p)
    q = norm.cdf((np.sqrt(rho) * _U - tau) / np.sqrt(1 - rho))
    return float(np.trapezoid(_PHI * binom.sf(Q - 1, n, q), _U))


def kritik_rho(n, p, hedef):
    from scipy.optimize import brentq
    return brentq(lambda r: p_analitik(n, p, r, n) - hedef, 1e-3, 0.999)


# ------------------------------------------------------------------ n_eff (Eş. 4)
def n_eff_tek(b, theta_c):
    """Açgözlü en-uzak-nokta sırası; eski makale_ispat.n_eff_tek ile aynı."""
    m = b.size
    if m <= 1:
        return float(m)
    D = aci_farki(b, b); np.fill_diagonal(D, np.inf)
    secili = [int(np.argmin(b))]; toplam = 1.0
    kalan = [i for i in range(m) if i not in secili]
    while kalan:
        en_k = D[np.ix_(kalan, secili)].min(axis=1)
        j = int(np.argmax(en_k))
        toplam += 1 - np.exp(-en_k[j] / theta_c)
        secili.append(kalan.pop(j))
    return toplam


def n_eff_tablo(b, theta_c):
    n = b.size; t = np.empty(1 << n)
    for c in range(1 << n):
        mask = np.array([(c >> i) & 1 for i in range(n)], bool)
        t[c] = n_eff_tek(b[mask], theta_c)
    return t


def n_eff_kurali(b, theta_c, kyp, kdg, butce):
    t = n_eff_tablo(np.asarray(b, float), theta_c)
    return butcede(t[kod(kyp)], t[kod(kdg)], butce)


# ------------------------------------------------------------------ ampirik çekirdek
KUTU = np.arange(0, 181, 15.0)


def ampirik(egri, ayni_model):
    """Ölçülen eğri (0..180°, 15° kutular) → ρ(Δθ) fonksiyonu. Bilgisiz (NaN) kutular
    komşulardan doğrusal; aynı modelde Δθ<15° için 15–30° eğiminden doğrusal
    ekstrapolasyon, [ρ(15°), 0.95] aralığına kırpılır."""
    y = np.array([np.nan if v is None else v for v in egri], float)
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        raise ValueError("eğri bilgisiz")
    y = np.interp(KUTU, KUTU[ok], y[ok])
    if ayni_model:
        y0 = np.clip(2 * y[1] - y[2], y[1], 0.95); y[0] = y0
    return lambda d, y=y: float(np.interp(d, KUTU, y))
