#!/usr/bin/env python3
"""D1 çoklu dedektör koşucusu — her kare BİR KEZ üretilir, M modele birden verilir.

Hakem revizyonu (2026-09-25) için:
  * K×K çapraz-model korelasyon matrisi (tek çift yerine M model)
  * bozulma ablation'ı: --mod ortak (orijinal, açıdan bağımsız bozulma) /
    --mod bagimsiz (bozulma ve piksel gürültüsü tohumu açıyı da içerir —
    farklı İHA kameralarının bağımsız gürültüsü/titremesi)

Çıktı: <cikti>/<mod>/<deneme_id>.npz
  conf[M, V, F]  float16  hedef sınıflarda (airplane, kite) en yüksek güven, yoksa 0
  ust[M, V, F]   int16    en güvenli tespitin COCO sınıf id'si, yoksa -1
  aci[V], modeller[M]
Eşik sonradan uygulanır (conf ≥ 0.40 = orijinal tespit_var) → eşik duyarlılığı da çıkar.

Kullanım:
  python d1_coklu.py --mod ortak    --modeller yolo11n.pt yolov8s.pt rtdetr-l.pt
  python d1_coklu.py --mod bagimsiz --modeller yolo11n.pt ...
"""
from __future__ import annotations
import argparse, dataclasses, os, sys, time
from multiprocessing import Pool
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d1_klip import ayar_yukle, denemeler, Uretici, bozulma_dizisi, AYAR_YOL

HEDEF_COCO = {4, 33}          # airplane, kite (COCO'da drone yok) — klip_ayar.yaml ile aynı küme
DUSUK_ESIK = 0.10             # NMS öncesi taban; 0.40 eşiği sonradan uygulanır (0.05 NMS darboğazı yarattı)

_U = None; _A = None


def _baslat(ayar_yol):
    global _U, _A
    _A = ayar_yukle(ayar_yol); _U = Uretici(_A)


def _klip(is_):
    """(deneme, görünüş indeksi, mod) → (V_idx, açı, kareler[F,H,W] uint8 gri)."""
    d, vi, mod, adim = is_
    if d.rol == "bos":
        n = None; aci = 0.0
    else:
        n = _U.nesneler(d)[vi]; aci = n.aci
    dd = d if mod == "ortak" else dataclasses.replace(d, id=f"{d.id}|aci={aci:g}")
    kareler = []
    for i, bz in enumerate(bozulma_dizisi(_A, dd)):
        if i % adim:
            continue                     # rastgele yürüyüş aynı kalır, yalnız çizim atlanır
        k, _ = _U.kare(dd, n, bz)
        kareler.append(k[:, :, 0])
    return vi, aci, np.stack(kareler)


def modelleri_yukle(adlar, imgsz, half):
    from ultralytics import YOLO, RTDETR
    ms = []
    for a in adlar:
        m = (RTDETR if "rtdetr" in a else YOLO)(a)
        ms.append(m)
    return ms


def gpu_toplu(kareler, half, imgsz=1280):
    """Gri kareler (F,H,W) uint8 → GPU'da (F,3,H',W') 0–1 tensör; ultralytics letterbox'ı
    ile aynı: genişlik 1280 değişmez, yükseklik 32'nin katına ORTALANMIŞ 114 dolgusuyla.
    Ön işleme bir kez yapılır, M model aynı tensörü kullanır (CPU darboğazı kalkar)."""
    import torch
    t = torch.from_numpy(kareler).cuda(non_blocking=True)
    if imgsz != t.shape[2]:                      # ör. RT-DETR doğal 640: önce ölçekle
        import torch.nn.functional as Fn
        s_ = imgsz / t.shape[2]
        t = Fn.interpolate(t[:, None].float(), size=(round(t.shape[1] * s_), imgsz), mode="bilinear",
                           align_corners=False, antialias=True)[:, 0].round().clamp(0, 255).to(torch.uint8)
    F, H, W = t.shape; Hp = (H + 31) // 32 * 32; ust_pay = (Hp - H) // 2
    x = torch.full((F, 3, Hp, W), 114, dtype=torch.uint8, device=t.device)
    x[:, :, ust_pay:ust_pay + H] = t[:, None]
    x = x.half() if half else x.float()
    return x / 255


def hedef_idleri(model):
    """COCO: airplane, kite; drone-eğitimli modeller: Drone. (bird hiçbir zaman hedef değil.)"""
    return {i for i, n in model.names.items() if n.lower() in ("airplane", "kite", "drone")}


def cikar(model, x, imgsz, half, batch):
    HEDEF = hedef_idleri(model)
    F = x.shape[0]; conf = np.zeros(F, np.float16); ust = np.full(F, -1, np.int16)
    for s in range(0, F, batch):
        rs = model.predict(x[s:s + batch], imgsz=(x.shape[2], x.shape[3]), conf=DUSUK_ESIK, verbose=False, half=half)
        for j, r in enumerate(rs):
            if len(r.boxes) == 0:
                continue
            c = r.boxes.conf.float().cpu().numpy(); k = r.boxes.cls.cpu().numpy().astype(int)
            ust[s + j] = k[c.argmax()]
            h = np.isin(k, list(HEDEF))
            if h.any():
                conf[s + j] = c[h].max()
    return conf, ust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", choices=["ortak", "bagimsiz"], required=True)
    ap.add_argument("--modeller", nargs="+", required=True)
    ap.add_argument("--cikti", default="kayit_coklu")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--isci", type=int, default=12)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--kare-adim", type=int, default=1, help="2 → yalnız çift kareler (0,2,..,58)")
    ap.add_argument("--sec", nargs="*")
    ap.add_argument("--ayar", default=AYAR_YOL, help="kayit/tam ile birebir: d1/klip_ayar_tam70.yaml")
    a = ap.parse_args()
    ayar = ayar_yukle(a.ayar); half = not a.fp32
    kok = os.path.join(a.cikti, a.mod); os.makedirs(kok, exist_ok=True)
    ms = modelleri_yukle(a.modeller, a.imgsz, half)
    with open(os.path.join(kok, "oturum.yaml"), "w") as f:
        yaml.safe_dump({"mod": a.mod, "modeller": a.modeller, "imgsz": a.imgsz, "half": half,
                        "dusuk_esik": DUSUK_ESIK, "kare_adim": a.kare_adim, "ayar": ayar, "hedef_siniflar": {a: sorted(hedef_idleri(m)) for a, m in zip(a.modeller, ms)},
                        "saat": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    u = Uretici(ayar)
    ds = denemeler(ayar)
    if a.sec:
        import fnmatch; ds = [d for d in ds if any(fnmatch.fnmatch(d.id, p) for p in a.sec)]
    t0 = time.time(); toplam = 0
    with Pool(a.isci, initializer=_baslat, initargs=(a.ayar,)) as pool:
        for k, d in enumerate(ds, 1):
            yol = os.path.join(kok, d.id + ".npz")
            if os.path.exists(yol):
                continue
            V = 1 if d.rol == "bos" else len(u.nesneler(d))
            F = len(range(0, d.kare_sayisi, a.kare_adim))
            conf = np.zeros((len(ms), V, F), np.float16); ust = np.full((len(ms), V, F), -1, np.int16)
            aci = np.zeros(V)
            for vi, ac, kareler in pool.imap_unordered(_klip, [(d, v, a.mod, a.kare_adim) for v in range(V)]):
                aci[vi] = ac
                x = gpu_toplu(kareler, half, a.imgsz)
                for mi, m in enumerate(ms):
                    conf[mi, vi], ust[mi, vi] = cikar(m, x, a.imgsz, half, a.batch)
                toplam += F
            np.savez_compressed(yol + ".tmp.npz", conf=conf, ust=ust, aci=aci, modeller=np.array(a.modeller),
                                kare_no=np.arange(0, d.kare_sayisi, a.kare_adim))
            os.replace(yol + ".tmp.npz", yol)
            hiz = toplam / (time.time() - t0)
            print(f"[{k}/{len(ds)}] {d.id}: V={V} tespit@.40 " +
                  " ".join(f"{(conf[i] >= .4).mean():.2f}" for i in range(len(ms))) +
                  f" | {hiz:.1f} kare/s", flush=True)
    print(f"bitti {toplam} kare {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
