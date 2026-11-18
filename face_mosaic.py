#!/usr/bin/env python3
"""写真・動画に写っている人の顔に自動でモザイクをかけるツール。

使い方:
    ./mosaic 写真.jpg
    ./mosaic 動画.mp4 --style blur
    ./mosaic フォルダ/            # フォルダ内の写真・動画をまとめて処理

出力は元ファイルと同じ場所に「元の名前_mosaic.拡張子」で保存される。
"""
import argparse
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # OpenCV 内部の警告を表示しない

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


class FaceDetector:
    def __init__(self, threshold, max_side, upscale=False):
        self.max_side = max_side
        self.upscale = upscale
        self.threshold = threshold
        self.net = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), threshold, 0.3, 5000)

    def _run(self, img, scale):
        h, w = img.shape[:2]
        resized = cv2.resize(img, (round(w * scale), round(h * scale))) if scale != 1 else img
        self.net.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = self.net.detect(resized)
        if faces is None:
            return [], []
        return [f[:4] / scale for f in faces], [float(f[-1]) for f in faces]

    def detect(self, img):
        """顔の矩形 (x, y, w, h) のリストを元画像の座標で返す。"""
        h, w = img.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        boxes, scores = self._run(img, scale)
        if self.upscale and scale * 2 * max(h, w) <= 4096:
            # 遠くの小さい顔は拡大した方が見つかりやすいので、2倍でもう一度探す
            b2, s2 = self._run(img, scale * 2)
            boxes += b2
            scores += s2
            # 同じ顔が両方で見つかった場合は1つにまとめる
            keep = cv2.dnn.NMSBoxes([list(map(float, b)) for b in boxes], scores, self.threshold, 0.3)
            boxes = [boxes[i] for i in np.array(keep).flatten()]
        return [tuple(b) for b in boxes]


def expand(box, margin, shape):
    """顔の枠を margin 倍だけ広げる（髪や耳まで隠すため）。上方向は少し多めに広げる。"""
    x, y, w, h = box
    H, W = shape[:2]
    dx, dy = w * margin, h * margin
    x0 = int(max(0, x - dx))
    y0 = int(max(0, y - dy * 1.4))
    x1 = int(min(W, x + w + dx))
    y1 = int(min(H, y + h + dy))
    return x0, y0, x1, y1


def overlaps(a, b):
    """2つの枠が同じ顔とみなせるくらい重なっているか。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    if iw <= 0 or ih <= 0:
        return False
    return iw * ih / min(aw * ah, bw * bh) > 0.3


def fill_gaps(history, center):
    """history[center] のフレームに使う枠を決める。

    そのフレームで見つかった顔に加えて、そのフレームだけ検出が漏れた顔は
    時間的に一番近いフレームの枠で補う。
    """
    chosen = list(history[center])
    for dist in range(1, len(history)):
        for i in (center - dist, center + dist):
            if 0 <= i < len(history):
                for b in history[i]:
                    if not any(overlaps(b, c) for c in chosen):
                        chosen.append(b)
    return chosen


def censor(img, boxes, style, strength, margin):
    for box in boxes:
        x0, y0, x1, y1 = expand(box, margin, img.shape)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = img[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        if style == "mosaic":
            # 顔の幅を strength 個のブロックに分割する
            block = max(2, rw // strength)
            tiny = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_AREA)
            img[y0:y1, x0:x1] = cv2.resize(tiny, (rw, rh), interpolation=cv2.INTER_NEAREST)
        elif style == "blur":
            k = max(3, (rw // 3) | 1)
            img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        else:  # fill
            img[y0:y1, x0:x1] = 0
    return img


def out_path(src, out_dir, ext=None):
    folder = Path(out_dir) if out_dir else src.parent
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{src.stem}_mosaic{ext or src.suffix}"


def read_image(path):
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image, ImageOps
        import pillow_heif

        pillow_heif.register_heif_opener()
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    # imread は EXIF の向きを自動で反映する。日本語パスでも読めるよう imdecode を使う
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def process_image(path, args):
    img = read_image(path)
    if img is None:
        print(f"  読み込めませんでした: {path}")
        return
    det = FaceDetector(args.threshold, max_side=2560, upscale=True)
    boxes = det.detect(img)
    censor(img, boxes, args.style, args.strength, args.margin)

    ext = ".jpg" if path.suffix.lower() in {".heic", ".heif"} else path.suffix
    dst = out_path(path, args.out, ext)
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        print(f"  保存に失敗しました: {dst}")
        return
    buf.tofile(str(dst))
    print(f"  顔 {len(boxes)} 個を処理 → {dst}")


def process_video(path, args):
    import imageio_ffmpeg

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  読み込めませんでした: {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        print(f"  フレームが読めませんでした: {path}")
        return
    h, w = first.shape[:2]

    dst = out_path(path, args.out, ".mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # 映像はパイプで渡し、音声は元ファイルからそのままコピーする（無ければ無音）
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
        "-i", str(path),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
        str(dst),
    ]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    det = FaceDetector(args.threshold, max_side=1920)
    # 検出漏れのフレームで顔が一瞬見えてしまうのを防ぐため、
    # 前後 hold フレーム分の検出結果もまとめてモザイクに使う
    hold = max(1, round(fps * args.hold))
    frames = deque()  # 出力待ちのフレーム
    history = deque(maxlen=hold * 2 + 1)  # 各フレームの検出結果
    face_frames = 0

    def emit():
        # 出力するフレームが history の何番目にあたるか
        center = len(history) - len(frames)
        img = frames.popleft()
        boxes = fill_gaps(history, center)
        enc.stdin.write(censor(img, boxes, args.style, args.strength, args.margin).tobytes())

    frame, n = first, 0
    while frame is not None:
        boxes = det.detect(frame)
        face_frames += bool(boxes)
        frames.append(frame)
        history.append(boxes)
        n += 1
        if len(frames) > hold:
            emit()
        if n % 30 == 0 or n == total:
            pct = f"{n * 100 // total}%" if total else f"{n}フレーム"
            print(f"\r  処理中... {pct}", end="", flush=True)
        ok, frame = cap.read()
        if not ok:
            frame = None
    # 残りのフレームを書き出す
    while frames:
        emit()

    cap.release()
    enc.stdin.close()
    enc.wait()
    print(f"\r  {n} フレーム中 {face_frames} フレームで顔を検出 → {dst}")


def collect(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.iterdir()
                              if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS and "_mosaic" not in f.stem)
        elif p.exists():
            yield p
        else:
            print(f"見つかりません: {p}")


def main():
    ap = argparse.ArgumentParser(description="写真・動画の顔に自動でモザイクをかけます")
    ap.add_argument("inputs", nargs="+", help="写真・動画ファイル、またはフォルダ")
    ap.add_argument("--style", choices=["mosaic", "blur", "fill"], default="mosaic",
                    help="mosaic=モザイク(既定) / blur=ぼかし / fill=黒塗り")
    ap.add_argument("--strength", type=int, default=8,
                    help="モザイクの細かさ。顔の横幅を何ブロックに分けるか（小さいほど粗い。既定 8）")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="顔の枠をどれだけ広げるか（既定 0.25 = 25%%）")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="顔とみなす確信度（下げると見逃しが減るが誤検出が増える。既定 0.6）")
    ap.add_argument("--hold", type=float, default=0.3,
                    help="動画で検出が途切れても前後何秒モザイクを残すか（既定 0.3）")
    ap.add_argument("--out", help="出力先フォルダ（省略時は元ファイルと同じ場所）")
    args = ap.parse_args()

    for path in collect(args.inputs):
        print(path.name)
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            process_image(path, args)
        elif ext in VIDEO_EXTS:
            process_video(path, args)
        else:
            print("  対応していない形式です")


if __name__ == "__main__":
    sys.exit(main())
