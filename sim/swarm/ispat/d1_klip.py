#!/usr/bin/env python3
"""D1 sentetik klip üretici — nesne görüntüsü + arka plan → dedektöre gidecek kareler.

Kareler DİSKE YAZILMAZ; her kare (deneme_id, kare_no) tohumundan deterministik
üretilir. Dedektör değişirse yeniden koşulur, "kayıt" tekrarlanmaz (protokol §4.3
ham video saklama kuralının sentetik karşılığı: tohum + nesne PNG'leri).

Kritik kural: bozulma parametreleri (kaydırma, ölçek, dönme, bulanıklık,
parlaklık, gürültü, arka plan kayması) AÇIDAN BAĞIMSIZDIR. Aynı denemede kare f,
bütün açılarda aynı bozulmayı görür; değişen yalnız nesnenin görünüşü. Analizdeki
kare-kare eşleme (d1_analiz.py §5.2) yalnız bu koşulda "iki ajan aynı anda farklı
açıdan bakıyor" anlamına gelir.

Kullanım:
  python3 d1_klip.py --manifest                 # deneme matrisi (sayı + liste)
  python3 d1_klip.py --onizle drone_01 --mesafe 6 --arka gok --isik parlak --aci 45
                                                # tek klip → mp4 + kontrol karesi
  python3 d1_klip.py --anahtar-test             # nesne maskelerini kontrol sayfasına bas
"""
from __future__ import annotations
import argparse, hashlib, os, re, sys
from dataclasses import dataclass
import numpy as np, cv2, yaml

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AYAR_YOL = os.path.join(KOK, "d1", "klip_ayar.yaml")


def ayar_yukle(yol=AYAR_YOL):
    with open(yol) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ nesneler
def beyaz_kenar_soy(bgr: np.ndarray, en_fazla=14) -> np.ndarray:
    """Hücre kırpması ızgara çizgisini içerebilir (karga: 3 kenarda ~3 px beyaz).
    Kenar şeridinin medyanı beyaza yakınsa soy."""
    for _ in range(en_fazla):
        h, w = bgr.shape[:2]; kes = False
        if np.median(bgr[0].min(1)) > 225: bgr = bgr[1:]; kes = True
        if np.median(bgr[-1].min(1)) > 225: bgr = bgr[:-1]; kes = True
        if np.median(bgr[:, 0].min(1)) > 225: bgr = bgr[:, 1:]; kes = True
        if np.median(bgr[:, -1].min(1)) > 225: bgr = bgr[:, :-1]; kes = True
        if not kes: break
    return bgr


def anahtarla(bgr: np.ndarray, ad: str = "") -> np.ndarray:
    """Düz gri fonu alfa maskesine çevir.
    Fon rengi: kenar piksellerinin MEDYANI, beyaza yakınlar (ızgara çizgisi
    kalıntısı) DIŞLANIR — ilk sürüm bunları dahil ediyordu, karga/insan
    hücrelerinde fon 'beyaz' sanılıp maske tümüyle dolmuş veya ters çıkmıştı.
    Gövde içindeki fon-renkli delikler (gri drone parçaları) doldurulur."""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # Fon örneği 4 px İÇERİDEN: en dış piksel ızgara çizgisinin JPEG halkalanmasını
    # taşıyor (karga: sol 182 / sağ 133, gerçek fon 163).
    ic = 4
    kenar_m = np.zeros((h, w), bool); kenar_m[[ic, -1 - ic], ic:-ic] = True; kenar_m[ic:-ic, [ic, -1 - ic]] = True
    kenar_m &= bgr.min(axis=2) < 200                    # beyaz ızgara kalıntısı dışarı
    if kenar_m.sum() < 20:
        kenar_m = np.zeros((h, w), bool); kenar_m[[0, -1], :] = True; kenar_m[:, [0, -1]] = True
    # Fon DÜZ DEĞİL olabilir (drone sayfasında köşeden köşeye 132→195 gradyan):
    # kenar piksellerine kanal başına ikinci dereceden yüzey oturt, farkı ona göre al.
    def taban(y, x):
        y = y / h; x = x / w
        return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)
    A_hepsi = taban(yy, xx).reshape(-1, 6)
    def yuzey(sec):
        A = taban(yy[sec], xx[sec])
        return np.stack([(A_hepsi @ np.linalg.lstsq(A, bgr[..., c][sec].astype(float), rcond=None)[0]).reshape(h, w)
                         for c in range(3)], -1)
    fon = yuzey(kenar_m)
    # Vinyetli fon (insan_03 sayfası): kenar yüzeyi merkezi ıskalar → iki tur
    # sağlam yeniden oturtma: artığı küçük TÜM pikseller fon örneği olur.
    esik = 22.0
    for _ in range(2):
        uzak = np.linalg.norm(bgr.astype(float) - fon, axis=2)
        ic_fon = (uzak < esik) & (bgr.min(axis=2) < 235)
        if ic_fon.sum() < 100:
            break
        fon = yuzey(ic_fon)
        uzak = np.linalg.norm(bgr.astype(float) - fon, axis=2)
        esik = float(max(22.0, 5.0 * np.median(uzak[ic_fon]) + 8.0))
    uzak = np.linalg.norm(bgr.astype(float) - fon, axis=2)
    beyaz = bgr.min(axis=2) > 235
    maske = ((uzak > esik) & ~beyaz).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, k)
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(maske)
    if n > 2:
        alan = ist[1:, cv2.CC_STAT_AREA]
        tut = np.zeros(n, bool); tut[1:] = alan >= max(20, 0.01 * alan.max())
        maske = np.where(tut[etiket], 255, 0).astype(np.uint8)
    # delik doldurma: kenardan taşma → ulaşılamayan fon = delik
    tasma = np.pad(255 - maske, 1, constant_values=255)
    cv2.floodFill(tasma, None, (0, 0), 0)
    maske = np.maximum(maske, tasma[1:-1, 1:-1])
    oran = (maske > 0).mean()
    if not 0.01 < oran < 0.85:
        raise SystemExit(f"anahtarlama şüpheli ({ad}): maske alanı %{100*oran:.0f}")
    return cv2.GaussianBlur(maske, (3, 3), 0)


@dataclass
class Nesne:
    ad: str            # ör. kus_karga
    aci: float         # nominal derece; açısız sınıflarda görünüş indeksi
    bgr: np.ndarray    # kırpılmış nesne
    alfa: np.ndarray   # 0–255


def nesne_yukle(ornek: str, acisiz=False) -> list[Nesne]:
    kl = os.path.join(KOK, "d1", "nesneler", ornek)
    cikti = []
    for dosya in sorted(os.listdir(kl)):
        if dosya.startswith("_") or not dosya.endswith(".png"):
            continue
        m = re.search(r"_(\d{3})\.png$", dosya) if not acisiz else re.search(r"_(\d+)\.png$", dosya)
        if not m:
            continue
        bgr = beyaz_kenar_soy(cv2.imread(os.path.join(kl, dosya), cv2.IMREAD_COLOR))
        alfa = anahtarla(bgr, dosya)
        ys, xs = np.where(alfa > 0)
        if len(xs) == 0:
            raise SystemExit(f"maske boş: {dosya}")
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cikti.append(Nesne(ornek, float(m.group(1)), bgr[y0:y1, x0:x1], alfa[y0:y1, x0:x1]))
    return cikti


# ------------------------------------------------------------------ deneme matrisi
@dataclass(frozen=True)
class Deneme:
    id: str
    sinif: str
    rol: str
    ornek: str | None
    mesafe_m: float | None
    arka: str
    isik: str
    kare_sayisi: int


def denemeler(ayar) -> list[Deneme]:
    out = []
    for sinif, s in ayar["siniflar"].items():
        for arka in ("gok", "kent"):
            for isik in ("parlak", "kapali"):
                if s["rol"] == "bos":
                    out.append(Deneme(f"bos_{arka}_{isik}", sinif, "bos", None, None, arka, isik,
                                      s.get("kare_sayisi", ayar["kare"]["kare_sayisi"])))
                    continue
                for ornek in s["ornekler"]:
                    for m in ayar["mesafeler_m"]:
                        mm = m * s.get("mesafe_carpani", 1)
                        out.append(Deneme(f"{ornek}_{int(mm):02d}m_{arka}_{isik}", sinif, s["rol"],
                                          ornek, float(mm), arka, isik, ayar["kare"]["kare_sayisi"]))
    return out


# ------------------------------------------------------------------ bozulma (açıdan bağımsız)
def _tohum(*parcalar) -> int:
    h = hashlib.sha256("|".join(map(str, parcalar)).encode()).digest()
    return int.from_bytes(h[:8], "little")


@dataclass
class Bozulma:
    dx: float; dy: float; olcek: float; donme: float; bulanik: float
    kazanc: float; gurultu: float; adx: int; ady: int


def bozulma_dizisi(ayar, d: Deneme) -> list[Bozulma]:
    """Denemenin tüm kareleri için bozulma; rastgele yürüyüş kareler arası
    süreklilik verir. Tohum yalnız deneme_id'den → açıdan bağımsız."""
    b = ayar["bozulma"]; rng = np.random.default_rng(_tohum("bozulma", d.id))
    n = d.kare_sayisi
    adim = rng.normal(0, b["kaydirma_adim_px"], (n, 2)).cumsum(0)
    adim = np.clip(adim, -b["kaydirma_sinir_px"], b["kaydirma_sinir_px"])
    olcek = np.exp(rng.normal(0, b["olcek_sigma"], n))
    donme = rng.normal(0, b["donme_sigma_derece"], n)
    bulanik = rng.uniform(*b["bulanik_px"], n)
    kazanc = rng.uniform(*b["parlaklik"][d.isik], n)
    gur = np.full(n, b["gurultu_sigma"][d.isik])
    ap = np.clip(rng.normal(0, b["arka_plan_kaydirma_px"] / 3, (n, 2)).cumsum(0),
                 -b["arka_plan_kaydirma_px"], b["arka_plan_kaydirma_px"]).astype(int)
    return [Bozulma(adim[i, 0], adim[i, 1], olcek[i], donme[i], bulanik[i], kazanc[i], gur[i],
                    ap[i, 0], ap[i, 1]) for i in range(n)]


# ------------------------------------------------------------------ kare üretimi
class Uretici:
    def __init__(self, ayar):
        self.a = ayar
        k = ayar["kare"]; self.W, self.H = k["genislik"], k["yukseklik"]
        self.f_px = (self.W / 2) / np.tan(np.radians(ayar["kamera"]["hfov_derece"] / 2))
        self._arka: dict[tuple, np.ndarray] = {}
        self._nesne: dict[str, list[Nesne]] = {}

    def arka_plan(self, arka, isik):
        if (arka, isik) not in self._arka:
            yol = os.path.join(KOK, self.a["arka_planlar"]["klasor"], self.a["arka_planlar"][arka][isik])
            im = cv2.imread(yol, cv2.IMREAD_COLOR)
            if im is None:
                raise SystemExit(f"arka plan yok: {yol}")
            # kaydırma payı için kareden büyük tut
            pay = self.a["bozulma"]["arka_plan_kaydirma_px"]
            im = cv2.resize(im, (self.W + 2 * pay, self.H + 2 * pay), interpolation=cv2.INTER_AREA)
            self._arka[(arka, isik)] = im
        return self._arka[(arka, isik)]

    def nesneler(self, d: Deneme) -> list[Nesne]:
        if d.ornek not in self._nesne:
            self._nesne[d.ornek] = nesne_yukle(d.ornek, self.a["siniflar"][d.sinif].get("acisiz", False))
        return self._nesne[d.ornek]

    def px_yukseklik(self, d: Deneme) -> float:
        return self.f_px * self.a["siniflar"][d.sinif]["boyut_m"] / d.mesafe_m

    def kare(self, d: Deneme, nesne: Nesne | None, bz: Bozulma) -> tuple[np.ndarray, tuple | None]:
        """Tek kare + nesnenin gerçek kutusu (x,y,w,h) (bearing hatası için)."""
        pay = self.a["bozulma"]["arka_plan_kaydirma_px"]
        ap = self.arka_plan(d.arka, d.isik)
        x0, y0 = pay + bz.adx, pay + bz.ady
        kare = ap[y0:y0 + self.H, x0:x0 + self.W].astype(np.float32)
        kutu = None
        if nesne is not None:
            hedef_h = self.px_yukseklik(d) * bz.olcek
            s = hedef_h / nesne.bgr.shape[0]
            M = cv2.getRotationMatrix2D((nesne.bgr.shape[1] / 2, nesne.bgr.shape[0] / 2), bz.donme, s)
            nw, nh = int(np.ceil(nesne.bgr.shape[1] * s * 1.3)) + 2, int(np.ceil(nesne.bgr.shape[0] * s * 1.3)) + 2
            M[0, 2] += nw / 2 - nesne.bgr.shape[1] / 2; M[1, 2] += nh / 2 - nesne.bgr.shape[0] / 2
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            ob = cv2.warpAffine(nesne.bgr, M, (nw, nh), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)).astype(np.float32)
            oa = cv2.warpAffine(nesne.alfa, M, (nw, nh), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32) / 255
            if bz.bulanik > 0.3:
                kk = int(bz.bulanik * 2) * 2 + 1
                ob = cv2.GaussianBlur(ob, (kk, kk), bz.bulanik); oa = cv2.GaussianBlur(oa, (kk, kk), bz.bulanik)
            cx, cy = self.W / 2 + bz.dx, self.H / 2 + bz.dy
            px, py = int(round(cx - nw / 2)), int(round(cy - nh / 2))
            # kareye kırp
            sx0, sy0 = max(0, -px), max(0, -py); ex0, ey0 = min(nw, self.W - px), min(nh, self.H - py)
            if ex0 > sx0 and ey0 > sy0:
                hedef = kare[py + sy0:py + ey0, px + sx0:px + ex0]
                a = oa[sy0:ey0, sx0:ex0, None]
                hedef[:] = hedef * (1 - a) + ob[sy0:ey0, sx0:ex0] * a
                ys, xs = np.where(oa[sy0:ey0, sx0:ex0] > 0.5)
                if len(xs):
                    kutu = (px + sx0 + xs.min(), py + sy0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        kare *= bz.kazanc
        rng = np.random.default_rng(_tohum("gurultu", d.id, id(bz) and 0, round(bz.dx, 3), round(bz.dy, 3)))
        kare += rng.normal(0, bz.gurultu, kare.shape).astype(np.float32)
        kare = np.clip(kare, 0, 255).astype(np.uint8)
        if self.a["kare"]["gri"]:
            g = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
            kare = cv2.merge([g, g, g])
        return kare, kutu

    def klip(self, d: Deneme, nesne: Nesne | None):
        """Üreteç: (kare_no, kare, gerçek kutu) — bozulma açıdan bağımsız."""
        for i, bz in enumerate(bozulma_dizisi(self.a, d)):
            k, kutu = self.kare(d, nesne, bz)
            yield i, k, kutu


# ------------------------------------------------------------------ CLI
def _manifest(ayar):
    ds = denemeler(ayar)
    from collections import Counter
    c = Counter(d.rol for d in ds)
    print(f"{len(ds)} deneme — {dict(c)}")
    top = 0
    u = Uretici(ayar)
    for d in ds:
        n_aci = 1 if d.rol == "bos" else len(u.nesneler(d))
        top += n_aci * d.kare_sayisi
    print(f"toplam kare: {top}  (~{top / 30 / 60:.0f} dk video eşdeğeri)")
    for d in ds[:6]: print(" ", d.id)
    print("  ...")


def _onizle(ayar, ornek, mesafe, arka, isik, aci, cikti):
    u = Uretici(ayar)
    d = next(x for x in denemeler(ayar) if x.ornek == ornek and x.arka == arka and x.isik == isik
             and abs(x.mesafe_m - mesafe * ayar["siniflar"][x.sinif].get("mesafe_carpani", 1)) < 1e-6)
    ns = u.nesneler(d)
    n = min(ns, key=lambda z: abs(z.aci - aci))
    os.makedirs(os.path.dirname(cikti) or ".", exist_ok=True)
    vw = cv2.VideoWriter(cikti, cv2.VideoWriter_fourcc(*"mp4v"), ayar["kare"]["fps"], (u.W, u.H))
    ilk = None
    for i, k, kutu in u.klip(d, n):
        vw.write(k)
        if ilk is None:
            ilk = k.copy()
            if kutu: cv2.rectangle(ilk, kutu[:2], (kutu[0] + kutu[2], kutu[1] + kutu[3]), (0, 255, 0), 1)
    vw.release()
    cv2.imwrite(cikti.replace(".mp4", "_kare0.png"), ilk)
    print(f"{d.id} açı {n.aci:.0f}: nesne {u.px_yukseklik(d):.0f} px yüksek → {cikti}")


def _anahtar_test(ayar, cikti):
    """Her örnekten 0° (veya ilk) görünüşü: orijinal | maske | dama fon üstüne bindirme."""
    satirlar = []
    for sinif, s in ayar["siniflar"].items():
        if s["rol"] == "bos": continue
        for ornek in s["ornekler"]:
            ns = nesne_yukle(ornek, s.get("acisiz", False))
            for n in ns[:: max(1, len(ns) // 4)][:4]:
                h, w = n.bgr.shape[:2]; sc = 160 / max(h, w)
                bgr = cv2.resize(n.bgr, (int(w * sc), int(h * sc))); alfa = cv2.resize(n.alfa, (int(w * sc), int(h * sc)))
                dama = ((np.indices(bgr.shape[:2]).sum(0) // 16) % 2 * 60 + 90).astype(np.uint8)
                dama = cv2.merge([dama] * 3).astype(np.float32); a = alfa[..., None] / 255
                bind = (dama * (1 - a) + bgr * a).astype(np.uint8)
                pano = np.full((170, 3 * 170, 3), 40, np.uint8)
                for j, im in enumerate([bgr, cv2.merge([alfa] * 3), bind]):
                    pano[5:5 + im.shape[0], j * 170 + 5:j * 170 + 5 + im.shape[1]] = im
                cv2.putText(pano, f"{ornek} {n.aci:.0f}", (5, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                satirlar.append(pano)
    sayfa = np.vstack([np.hstack(satirlar[i:i + 4]) if len(satirlar[i:i + 4]) == 4 else
                       np.hstack(satirlar[i:i + 4] + [np.zeros_like(satirlar[0])] * (4 - len(satirlar[i:i + 4])))
                       for i in range(0, len(satirlar), 4)])
    os.makedirs(os.path.dirname(cikti) or ".", exist_ok=True)
    cv2.imwrite(cikti, sayfa); print("→", cikti)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ayar", default=AYAR_YOL)
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--onizle", metavar="ORNEK")
    ap.add_argument("--mesafe", type=float, default=6); ap.add_argument("--arka", default="gok")
    ap.add_argument("--isik", default="parlak"); ap.add_argument("--aci", type=float, default=0)
    ap.add_argument("--cikti", default=os.path.join(KOK, "d1", "onizleme", "klip.mp4"))
    ap.add_argument("--anahtar-test", action="store_true")
    a = ap.parse_args(); ayar = ayar_yukle(a.ayar)
    if a.manifest: _manifest(ayar)
    if a.anahtar_test: _anahtar_test(ayar, os.path.join(KOK, "d1", "onizleme", "anahtar_test.png"))
    if a.onizle: _onizle(ayar, a.onizle, a.mesafe, a.arka, a.isik, a.aci, a.cikti)
    if not (a.manifest or a.onizle or a.anahtar_test): ap.print_help()
