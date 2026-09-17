#!/usr/bin/env python3
"""
blur_faces.py — 動画内の顔を自動検出して自然なぼかしを入れるツール

パイプライン:
  ffmpeg(VideoToolboxでハードウェアデコード) -> このスクリプト(顔検出+ぼかし) -> ffmpeg(VideoToolboxでエンコード+元音声を再結合)

想定環境: Apple Silicon Mac (M3 / 8GB RAM) / 4K, 40〜60分の長尺素材
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"

# ---------- 動画情報の取得 ----------

def ffprobe_info(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip() != ""
    return {"width": width, "height": height, "fps": fps, "duration": duration, "has_audio": has_audio}


# ---------- 軽量トラッカー(検出の抜け・チラつきを補う) ----------

class Track:
    __slots__ = ("cx", "cy", "w", "h", "missed", "id")
    _next_id = 0

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        self.missed = 0
        self.id = Track._next_id
        Track._next_id += 1

    def box(self):
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h)


class FaceTracker:
    """フレーム間で同一人物のボックスを緩やかに追従・平滑化し、
    数フレーム検出が抜けても位置を保持することでチラつきを防ぐ。"""

    def __init__(self, max_missed=10, smooth=0.5, match_dist_ratio=0.8):
        self.tracks: list[Track] = []
        self.max_missed = max_missed
        self.smooth = smooth  # 大きいほど新しい検出に素早く追従
        self.match_dist_ratio = match_dist_ratio

    def update(self, detections):
        """detections: list of (cx, cy, w, h)"""
        unmatched = list(range(len(detections)))
        for tr in self.tracks:
            best_i, best_d = -1, None
            for i in unmatched:
                cx, cy, w, h = detections[i]
                d = ((cx - tr.cx) ** 2 + (cy - tr.cy) ** 2) ** 0.5
                thresh = max(tr.w, w) * self.match_dist_ratio
                if d < thresh and (best_d is None or d < best_d):
                    best_i, best_d = i, d
            if best_i >= 0:
                cx, cy, w, h = detections[best_i]
                a = self.smooth
                tr.cx = tr.cx * (1 - a) + cx * a
                tr.cy = tr.cy * (1 - a) + cy * a
                tr.w = tr.w * (1 - a) + w * a
                tr.h = tr.h * (1 - a) + h * a
                tr.missed = 0
                unmatched.remove(best_i)
            else:
                tr.missed += 1

        for i in unmatched:
            cx, cy, w, h = detections[i]
            self.tracks.append(Track(cx, cy, w, h))

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t.box() for t in self.tracks]

    def reset(self):
        self.tracks = []


def is_scene_cut(prev_thumb, cur_thumb, threshold):
    """フレーム間の変化量からカット(シーン切り替わり)を検知する。
    小さいサムネイルでの平均輝度差なので計算コストはごく小さい。"""
    if prev_thumb is None:
        return False
    diff = float(np.mean(np.abs(cur_thumb.astype(np.int16) - prev_thumb.astype(np.int16))))
    return diff > threshold


# ---------- ぼかし処理 ----------

def blur_regions(frame, boxes, pad_ratio, strength):
    """boxes: list of (x, y, w, h) in full-resolution pixel coords.
    楕円形+ソフトエッジで馴染むように合成する。
    strength: 顔サイズに対するぼかし半径の割合(0-1程度)。顔の大小によらず
    見た目の"ぼけ具合"が揃うよう、絶対px数ではなく比率でカーネルを決める。"""
    h_img, w_img = frame.shape[:2]
    out = frame
    for (x, y, w, h) in boxes:
        pad_w, pad_h = w * pad_ratio, h * pad_ratio
        x0 = max(0, int(x - pad_w))
        y0 = max(0, int(y - pad_h * 1.15))  # 上方向(髪の生え際)は少しだけ多めに
        x1 = min(w_img, int(x + w + pad_w))
        y1 = min(h_img, int(y + h + pad_h * 1.05))  # 下方向(顎)もわずかに多めに
        if x1 <= x0 or y1 <= y0:
            continue

        roi = out[y0:y1, x0:x1]
        rw, rh = roi.shape[1], roi.shape[0]
        if rw < 2 or rh < 2:
            continue

        # 顔サイズに比例したガウスぼかし(縮小拡大はしない=ブロック状にならず滑らか)
        k = int(min(rw, rh) * strength)
        k = min(k, min(rw, rh) - 1 if min(rw, rh) % 2 == 0 else min(rw, rh))
        k = max(5, k if k % 2 == 1 else k + 1)
        blurred = cv2.GaussianBlur(roi, (k, k), 0)

        # 楕円ソフトマスクで自然に合成(境界線が見えないように)
        mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 1.0, -1)
        feather = max(3, min(rw, rh) // 6) | 1
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        mask3 = mask[:, :, None]

        out[y0:y1, x0:x1] = (blurred * mask3 + roi * (1 - mask3)).astype(np.uint8)
    return out


# ---------- メイン処理 ----------

def main():
    ap = argparse.ArgumentParser(description="動画内の顔を自動検出してぼかすツール")
    ap.add_argument("input", help="入力動画パス")
    ap.add_argument("output", help="出力動画パス")
    ap.add_argument("--start", type=float, default=0.0, help="開始位置(秒)。テスト・分割処理用")
    ap.add_argument("--duration", type=float, default=None, help="処理する長さ(秒)。省略時は最後まで")
    ap.add_argument("--detect-width", type=int, default=1280, help="検出用に縮小する横幅(速度と精度のバランス)")
    ap.add_argument("--confidence", type=float, default=0.5, help="検出の信頼度しきい値(0-1)")
    ap.add_argument("--pad", type=float, default=0.22, help="顔ボックスの余白率(大きいほど広く隠す)")
    ap.add_argument("--strength", type=float, default=0.5,
                     help="ぼかしの強さ。顔サイズに対する割合(0-1)。大きいほど強い")
    ap.add_argument("--max-missed", type=int, default=10, help="追従を保持する最大フレーム数(チラつき対策)")
    ap.add_argument("--cut-threshold", type=float, default=18.0,
                     help="シーンカット検知の感度(小さいほど敏感。0でカット検知を無効化)")
    ap.add_argument("--bitrate", type=int, default=100,
                     help="出力ビットレート(Mbps)。元素材と同等以上を目安に(4K高品質素材なら80〜120程度)")
    ap.add_argument("--preview", action="store_true", help="処理結果を画面表示しながら実行(デバッグ用、遅くなる)")
    args = ap.parse_args()

    info = ffprobe_info(args.input)
    width, height, fps = info["width"], info["height"], info["fps"]
    duration = args.duration if args.duration else max(0.0, info["duration"] - args.start)
    total_frames_est = int(duration * fps)

    print(f"[info] {width}x{height} @ {fps:.2f}fps, 対象区間 {duration/60:.1f} 分 "
          f"(開始 {args.start:.0f}秒), 推定フレーム数 {total_frames_est}")

    detect_scale = args.detect_width / width
    detect_h = int(height * detect_scale)

    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (args.detect_width, detect_h),
        score_threshold=args.confidence, nms_threshold=0.3, top_k=5000,
    )
    tracker = FaceTracker(max_missed=args.max_missed)

    # --- ffmpeg decode: 生フレームをVideoToolboxでハードウェアデコードして受け取る ---
    decode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "videotoolbox",
        "-ss", str(args.start), "-i", args.input,
    ]
    if args.duration:
        decode_cmd += ["-t", str(args.duration)]
    decode_cmd += [
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-",
    ]

    # --- ffmpeg encode: 処理済みフレームをVideoToolboxでエンコードし、元音声を再結合 ---
    encode_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
    ]
    if info["has_audio"]:
        encode_cmd += ["-ss", str(args.start), "-i", args.input]
        if args.duration:
            encode_cmd += ["-t", str(args.duration)]
        encode_cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "copy"]
    bitrate_bps = args.bitrate * 1_000_000
    encode_cmd += [
        "-c:v", "h264_videotoolbox",
        "-b:v", str(bitrate_bps), "-maxrate", str(int(bitrate_bps * 1.5)),
        "-bufsize", str(bitrate_bps * 2),
        "-pix_fmt", "yuv420p", args.output,
    ]

    frame_bytes = width * height * 3
    decode_proc = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 2)
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    cut_count = 0
    prev_thumb = None
    t0 = time.time()
    try:
        while True:
            raw = decode_proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()

            small = cv2.resize(frame, (args.detect_width, detect_h), interpolation=cv2.INTER_AREA)

            if args.cut_threshold > 0:
                thumb = cv2.cvtColor(cv2.resize(small, (64, 36)), cv2.COLOR_BGR2GRAY)
                if is_scene_cut(prev_thumb, thumb, args.cut_threshold):
                    tracker.reset()  # カットをまたいでぼかしが居座らないようにリセット
                    cut_count += 1
                prev_thumb = thumb

            _, faces = detector.detect(small)

            detections = []
            if faces is not None:
                for f in faces:
                    x, y, w, h = f[0], f[1], f[2], f[3]
                    cx = (x + w / 2) / detect_scale
                    cy = (y + h / 2) / detect_scale
                    detections.append((cx, cy, w / detect_scale, h / detect_scale))

            boxes = tracker.update(detections)  # 既に(x, y, w, h)の左上基準で返る
            frame = blur_regions(frame, boxes, args.pad, args.strength)

            if args.preview:
                cv2.imshow("preview (qで終了)", cv2.resize(frame, (960, int(960 * height / width))))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            encode_proc.stdin.write(frame.tobytes())
            frame_idx += 1

            if frame_idx % int(fps * 5) == 0:
                elapsed = time.time() - t0
                fps_actual = frame_idx / elapsed
                remain = (total_frames_est - frame_idx) / max(fps_actual, 0.01)
                print(f"\r[progress] {frame_idx}/{total_frames_est} frames "
                      f"({fps_actual:.1f} fps処理, 残り約{remain/60:.1f}分)", end="", flush=True)
    finally:
        decode_proc.stdout.close()
        decode_proc.wait()
        encode_proc.stdin.close()
        encode_proc.wait()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"\n[done] {frame_idx}フレーム処理完了 -> {args.output}  "
          f"(所要 {(time.time()-t0)/60:.1f}分, カット検知 {cut_count}回)")


if __name__ == "__main__":
    main()
