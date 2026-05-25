
import argparse
from pathlib import Path
import cv2

SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in sorted(root.rglob("*")):
        if p.suffix not in SUPPORTED_EXTS:
            continue
        cap = cv2.VideoCapture(str(p))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"WARNING: could not read first frame of {p}")
            continue
        h, w = frame.shape[:2]
        # draw default central box (50% size) to help you copy coordinates
        cw, ch = int(w*0.5), int(h*0.5)
        x = (w - cw)//2
        y = (h - ch)//2
        frame_box = frame.copy()
        cv2.rectangle(frame_box, (x,y), (x+cw,y+ch), (255,255,255), 2)
        out = outdir / f"{p.stem}_firstframe.png"
        cv2.imwrite(str(out), frame_box)
        print(f"Saved {out}   (default ROI x={x}, y={y}, w={cw}, h={ch})")

if __name__ == "__main__":
    main()
