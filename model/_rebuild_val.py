"""Rebuild the BUU val split (seed=42) and compute real per-level metrics."""
import os, re, glob, json, shutil, random
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

BUU_ROOT = "./BUU-LSPINE-2000"
VIEWS = ["AP", "LA"]
CLASS_NAMES = ["L1", "L2", "L3", "L4", "L5"]
VAL_RATIO, SEED, IMGSZ = 0.15, 42, 640
ROOT = os.path.abspath("yolo_ap_la_seg")
WEIGHTS = "runs/segment/ap_seg/l1_l5_yolo26m_ap_la/weights/best.pt"


def view_dirs(v):
    base = os.path.join(BUU_ROOT, v)
    return os.path.join(base, "images"), os.path.join(base, "labels")


def csv_rows(p):
    out = []
    for line in open(p):
        nums = re.findall(r"-?\d+\.?\d*", line)
        if len(nums) >= 4:
            out.append([float(v) for v in nums[:4]])
    return out


def order_quad(pts):
    pts = sorted(pts, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[1], bot[0]]


def parse_polys(p, n=5, lpv=2):
    rows = csv_rows(p)
    if len(rows) < lpv * n:
        return []
    rows = rows[:lpv * n]
    verts = []
    for i in range(n):
        pts = []
        for j in range(lpv):
            xL, yL, xR, yR = rows[i * lpv + j]
            pts += [(xL, yL), (xR, yR)]
        q = order_quad(pts)
        verts.append((sum(pt[1] for pt in q) / 4.0, q))
    verts.sort(key=lambda v: v[0])
    return [q for _, q in verts]


def img_wh(path):
    with Image.open(path) as im:
        return im.size


def build():
    pairs = []
    for view in VIEWS:
        img_dir, lbl_dir = view_dirs(view)
        for lp in sorted(glob.glob(os.path.join(lbl_dir, "*.csv"))):
            stem = os.path.splitext(os.path.basename(lp))[0]
            ip = next((os.path.join(img_dir, stem + e) for e in (".jpg", ".jpeg", ".png")
                       if os.path.exists(os.path.join(img_dir, stem + e))), None)
            if ip and len(parse_polys(lp)) == len(CLASS_NAMES):
                pairs.append((view, ip, lp))
    random.Random(SEED).shuffle(pairs)
    n_val = int(len(pairs) * VAL_RATIO)
    split = {"val": pairs[:n_val], "train": pairs[n_val:]}
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = os.path.join(ROOT, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    def emit(item):
        sp, view, ip, lp = item
        ext = os.path.splitext(ip)[1]
        stem = f"{view}_" + os.path.splitext(os.path.basename(ip))[0]
        W, H = img_wh(ip)
        shutil.copy(ip, os.path.join(ROOT, "images", sp, stem + ext))
        with open(os.path.join(ROOT, "labels", sp, stem + ".txt"), "w") as f:
            for cid, q in enumerate(parse_polys(lp)):
                coords = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in q)
                f.write(f"{cid} {coords}\n")

    tasks = [(sp, view, ip, lp) for sp, pl in split.items() for (view, ip, lp) in pl]
    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
        list(ex.map(emit, tasks))

    yaml_path = os.path.join(ROOT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {ROOT}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    return yaml_path, len(split["train"]), len(split["val"])


def poly_to_mask(poly_xy, H, W):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.asarray(poly_xy, np.int32)], 1)
    return m


def dice(a, b):
    a = a > 0; b = b > 0
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def main():
    yaml_path = os.path.join(ROOT, "data.yaml")
    if os.path.exists(yaml_path) and len(glob.glob(os.path.join(ROOT, "images", "val", "*"))) > 0:
        nva = len(glob.glob(os.path.join(ROOT, "images", "val", "*")))
        ntr = len(glob.glob(os.path.join(ROOT, "images", "train", "*")))
        print(f">> reusing dataset  train={ntr}  val={nva}", flush=True)
    else:
        print(">> building dataset (seed=42) ...", flush=True)
        yaml_path, ntr, nva = build()
        print(f"   train={ntr}  val={nva}", flush=True)

    from ultralytics import YOLO
    seg = YOLO(WEIGHTS)

    print(">> official seg.val() ...", flush=True)
    res = seg.val(data=yaml_path, imgsz=IMGSZ, split="val", verbose=False, plots=False, workers=0)

    names = res.names
    mp = res.seg
    per = {}
    for i, c in enumerate(mp.ap_class_index):
        per[names[c]] = {"mask_mAP50": float(mp.ap50[i]), "mask_mAP50_95": float(mp.ap[i]),
                         "precision": float(mp.p[i]), "recall": float(mp.r[i])}
    summary = {"mask_mAP50_all": float(mp.map50), "mask_mAP50_95_all": float(mp.map),
               "box_mAP50_all": float(res.box.map50), "per_class": per, "n_val": nva}

    print(">> computing per-class mask Dice on val ...", flush=True)
    val_imgs = sorted(glob.glob(os.path.join(ROOT, "images", "val", "*")))
    dices = {c: [] for c in range(len(CLASS_NAMES))}
    for ip in val_imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(ROOT, "labels", "val", stem + ".txt")
        if not os.path.exists(lp):
            continue
        img = cv2.imread(ip)
        H, W = img.shape[:2]
        gt = {}
        for line in open(lp):
            parts = line.split()
            cid = int(parts[0])
            xy = np.array(parts[1:], float).reshape(-1, 2) * [W, H]
            gt[cid] = poly_to_mask(xy, H, W)
        r = seg.predict(img, conf=0.25, imgsz=IMGSZ, verbose=False)[0]
        pred = {}
        if r.masks is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for c in range(len(CLASS_NAMES)):
                idx = [i for i in range(len(cls)) if cls[i] == c]
                if idx:
                    best = max(idx, key=lambda i: conf[i])
                    pred[c] = poly_to_mask(r.masks.xy[best], H, W)
        for c, gm in gt.items():
            pm = pred.get(c, np.zeros_like(gm))
            dices[c].append(dice(pm, gm))

    per_dice = {}
    for c, name in enumerate(CLASS_NAMES):
        arr = np.array(dices[c], float)
        per_dice[name] = {"n": int(arr.size),
                          "mask_dice_mean": float(arr.mean()) if arr.size else None,
                          "mask_dice_median": float(np.median(arr)) if arr.size else None}
    summary["per_class_dice"] = per_dice

    os.makedirs("l4_v7_output/metrics", exist_ok=True)
    with open("l4_v7_output/metrics/buu_val_per_level.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== BUU VAL - per-level results =====")
    print(f"n_val = {nva}   overall mask mAP50 = {summary['mask_mAP50_all']:.3f}   mAP50-95 = {summary['mask_mAP50_95_all']:.3f}")
    print(f"{'level':6s} {'Dice(mean)':>11s} {'Dice(med)':>10s} {'mAP50':>7s} {'mAP50-95':>9s} {'P':>6s} {'R':>6s}")
    for name in CLASS_NAMES:
        d = per_dice[name]; m = per.get(name, {})
        print(f"{name:6s} {d['mask_dice_mean']:11.3f} {d['mask_dice_median']:10.3f} "
              f"{m.get('mask_mAP50', float('nan')):7.3f} {m.get('mask_mAP50_95', float('nan')):9.3f} "
              f"{m.get('precision', float('nan')):6.3f} {m.get('recall', float('nan')):6.3f}")
    print("\nsaved -> l4_v7_output/metrics/buu_val_per_level.json")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
