#!/usr/bin/env python3
"""
paddle_ocr_bridge.py — persistent PaddleOCR worker for PgsToSrtPlus.

The C# host spawns this script once.  The model is loaded on startup and
then reused for every image, avoiding per-image process start overhead.

Protocol: newline-delimited JSON over stdin / stdout.

  Startup — C# sends one config line then waits for the ready signal:
    {"model_name": "PP-OCRv5_server_rec", "device": "cpu",
     "italic_threshold": 3.0}

  Ready signal — Python -> C#:
    {"ready": true}
    or on failure:
    {"error": "..."}

  Per-image OCR request — C# -> Python:
    {"image": "/absolute/path/to/line.png"}

  Per-image OCR response — Python -> C#:
    {"text": "joined text", "score": 0.97,
     "segments": [{"text": "YODA:", "confidence": 0.98,
                    "bbox": [x1, y1, x2, y2]}, ...]}
    or:
    {"error": "..."}

  Italic detection request — C# -> Python:
    {"italic_detect": "/absolute/path/to/image.png"}

  Italic detection response — Python -> C#:
    {"angle": -7.3, "is_italic": true}
    or:
    {"error": "..."}

  Shutdown — C# -> Python:
    {"quit": true}
"""

import json
import os
import sys

import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


# -- Shear projection italic detection ----------------------------------------

def _shear_projection_variance(img_gray, angle_deg, pad):
    """Compute variance of the vertical projection after shearing rows.

    The canvas is padded horizontally by `pad` pixels on each side so that
    content is never clipped at extreme shear angles — without this,
    variance rises monotonically toward the angle-range boundaries and
    produces false-positive italic detections on upright text.
    """
    h, w = img_gray.shape
    if h == 0 or w == 0:
        return 0.0
    shear = np.tan(np.radians(angle_deg))
    padded_w = w + 2 * pad
    result = np.zeros((h, padded_w), dtype=img_gray.dtype)
    for y in range(h):
        shift = int(shear * y) + pad
        if 0 <= shift <= padded_w - w:
            result[y, shift:shift + w] = img_gray[y, :]
    profile = result.sum(axis=0).astype(np.float64)
    return float(np.var(profile))


def _detect_italic_angle(img_gray, angle_range=(-20, 20), steps=81):
    """Sweep shear angles and return the one with peak projection variance."""
    h, _ = img_gray.shape
    max_angle = max(abs(angle_range[0]), abs(angle_range[1]))
    pad = int(np.tan(np.radians(max_angle)) * max(h - 1, 0)) + 1
    angles = np.linspace(angle_range[0], angle_range[1], steps)
    variances = [_shear_projection_variance(img_gray, a, pad) for a in angles]
    best_angle = float(angles[np.argmax(variances)])
    return best_angle


def _binarize(img_gray, threshold=128):
    """Invert + threshold: text pixels become white (255), background black (0).

    Raw grayscale projections are dominated by the white background, masking
    the text signal and causing italic angles to be undetectable.  Binarizing
    first ensures only text strokes contribute to the projection profile.
    """
    return ((img_gray < threshold) * 255).astype(np.uint8)


def _classify_italic(img_gray, threshold):
    """Return (is_italic, angle) for a grayscale text image."""
    binary = _binarize(img_gray)
    angle = _detect_italic_angle(binary)
    return abs(angle) > threshold, angle


# -- Text extraction helpers (for recognition-only fallback) -------------------

def _extract_text_score(obj) -> tuple[str, float]:
    """Extract (text, score) from a PaddleOCR TextRecognition predict() result."""
    if obj is None:
        return "", 0.0
    if isinstance(obj, str):
        return obj.strip(), 0.0

    if isinstance(obj, dict):
        if "rec_texts" in obj:
            texts = obj.get("rec_texts") or []
            scores = obj.get("rec_scores") or []
            if isinstance(texts, (list, tuple)):
                text = " ".join(str(t).strip() for t in texts if t is not None and str(t).strip())
            else:
                text = str(texts).strip()
            try:
                score = float(sum(float(s) for s in scores) / len(scores)) if scores else 0.0
            except Exception:
                score = 0.0
            return text, score
        if "res" in obj and isinstance(obj["res"], dict):
            res = obj["res"]
            return str(res.get("rec_text", "") or "").strip(), float(res.get("rec_score", 0.0) or 0.0)
        if "rec_text" in obj:
            return str(obj.get("rec_text", "") or "").strip(), float(obj.get("rec_score", 0.0) or 0.0)
        if "text" in obj:
            return str(obj.get("text", "") or "").strip(), float(obj.get("score", 0.0) or 0.0)

    for attr in ("to_dict", "to_json"):
        val = getattr(obj, attr, None)
        if callable(val):
            try:
                val = val()
                if isinstance(val, str):
                    val = json.loads(val)
                if isinstance(val, dict):
                    return _extract_text_score(val)
            except Exception:
                pass

    return str(obj).strip(), 0.0


# -- Main ---------------------------------------------------------------------

def main() -> None:
    # -- Startup: read config from first line -----------------------------------
    try:
        init_raw = sys.stdin.readline()
        cfg = json.loads(init_raw) if init_raw.strip() else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid startup JSON: {exc}"}), flush=True)
        return

    model_name = cfg.get("model_name", "PP-OCRv5_server_rec")
    device = cfg.get("device", "cpu")
    italic_threshold = float(cfg.get("italic_threshold", 3.0))
    det_unclip_ratio = float(cfg.get("det_unclip_ratio", 5.0))

    if device.lower().startswith("gpu"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["FLAGS_use_cuda"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        paddle_device = f"gpu:{gpu_id}"
    else:
        os.environ["FLAGS_use_cuda"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        paddle_device = "cpu"

    # -- Load models (once) -----------------------------------------------------
    try:
        import paddle
        paddle.set_device(paddle_device)

        from paddleocr import PaddleOCR, TextRecognition

        # Full pipeline for segment detection + recognition
        ocr = PaddleOCR(
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang="en",
            device=paddle_device,
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name=model_name,
            text_det_unclip_ratio=det_unclip_ratio,
        )

        # Recognition-only model for VLM fallback path
        rec = TextRecognition(model_name=model_name, device=paddle_device)

        print(json.dumps({"ready": True}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"model load failed: {exc}"}), flush=True)
        return

    # -- Request loop -----------------------------------------------------------
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid JSON request"}), flush=True)
            continue

        if req.get("quit"):
            break

        # --- Word segmentation request (projection profile) ---
        if "word_segment" in req:
            image_path = req["word_segment"]
            try:
                from PIL import Image
                img = Image.open(image_path).convert("L")
                img_gray = np.array(img)
                binary = _binarize(img_gray)
                col_sums = binary.sum(axis=0)

                words = []
                in_word = False
                start = 0
                for x in range(len(col_sums)):
                    if col_sums[x] > 0 and not in_word:
                        start = x
                        in_word = True
                    elif col_sums[x] == 0 and in_word:
                        words.append([int(start), int(x)])
                        in_word = False
                if in_word:
                    words.append([int(start), int(len(col_sums))])

                print(json.dumps({"words": words}), flush=True)
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)
            continue

        # --- Italic detection request ---
        if "italic_detect" in req:
            image_path = req["italic_detect"]
            try:
                from PIL import Image
                img = Image.open(image_path).convert("L")
                img_gray = np.array(img)
                is_italic, angle = _classify_italic(img_gray, italic_threshold)
                print(json.dumps({
                    "angle": round(angle, 2),
                    "is_italic": is_italic,
                }), flush=True)
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)
            continue

        # --- Recognition-only request (for VLM fallback path) ---
        if req.get("recognize_only"):
            image_path = req.get("image")
            if not image_path:
                print(json.dumps({"error": "missing 'image' field"}), flush=True)
                continue
            try:
                out = rec.predict(input=image_path, batch_size=1)
                text, score = _extract_text_score(out[0] if out else None)
                print(json.dumps({"text": text, "score": score}), flush=True)
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)
            continue

        # --- Full OCR request (detection + recognition) ---
        image_path = req.get("image")
        if not image_path:
            print(json.dumps({"error": "missing 'image' field"}), flush=True)
            continue

        try:
            results = ocr.predict(image_path)

            # Collect segments from result
            segments = []
            texts = []
            scores = []

            for r in results:
                rec_texts = r.get("rec_texts", [])
                rec_scores = r.get("rec_scores", [])
                rec_boxes = r.get("rec_boxes")

                if rec_boxes is not None and hasattr(rec_boxes, "tolist"):
                    rec_boxes = rec_boxes.tolist()

                for i, seg_text in enumerate(rec_texts):
                    seg_conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                    bbox = rec_boxes[i] if rec_boxes and i < len(rec_boxes) else [0, 0, 0, 0]

                    segments.append({
                        "text": seg_text,
                        "confidence": round(seg_conf, 4),
                        "bbox": [int(v) for v in bbox],
                    })
                    texts.append(seg_text)
                    scores.append(seg_conf)

            if not segments:
                # Detection found nothing — fall back to recognition-only
                out = rec.predict(input=image_path, batch_size=1)
                text, score = _extract_text_score(out[0] if out else None)
                print(json.dumps({
                    "text": text,
                    "score": score,
                    "segments": [],
                }), flush=True)
                continue

            joined_text = " ".join(texts)
            avg_score = sum(scores) / len(scores) if scores else 0.0

            print(json.dumps({
                "text": joined_text,
                "score": round(avg_score, 4),
                "segments": segments,
            }), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
