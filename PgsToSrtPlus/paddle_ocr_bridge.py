#!/usr/bin/env python3
"""
paddle_ocr_bridge.py — persistent PaddleOCR worker for PgsToSrtPlus.

The C# host spawns this script once.  The model is loaded on startup and
then reused for every image, avoiding per-image process start overhead.

Protocol: newline-delimited JSON over stdin / stdout.

  Startup — C# sends one config line then waits for the ready signal:
    {"model_name": "PP-OCRv5_server_rec", "device": "cpu"}

  Ready signal — Python → C#:
    {"ready": true}
    or on failure:
    {"error": "..."}

  Per-image request — C# → Python:
    {"image": "/absolute/path/to/line.png"}

  Per-image response — Python → C#:
    {"text": "recognized text", "score": 0.97}
    or:
    {"error": "..."}

  Shutdown — C# → Python:
    {"quit": true}
"""

import json
import os
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


# ── Text extraction helpers ────────────────────────────────────────────────────

def _extract_text_score(obj) -> tuple[str, float]:
    """Extract (text, score) from a PaddleOCR TextRecognition predict() result."""
    if obj is None:
        return "", 0.0
    if isinstance(obj, str):
        return obj.strip(), 0.0

    # Dict formats used by different PaddleOCR versions.
    if isinstance(obj, dict):
        if "rec_texts" in obj:
            texts  = obj.get("rec_texts") or []
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

    # Object with to_dict / to_json method (PaddleX result objects).
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Startup: read config from first line ──────────────────────────────────
    try:
        init_raw = sys.stdin.readline()
        cfg = json.loads(init_raw) if init_raw.strip() else {}
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid startup JSON: {exc}"}), flush=True)
        return

    model_name = cfg.get("model_name", "PP-OCRv5_server_rec")
    device     = cfg.get("device", "cpu")

    if device.lower().startswith("gpu"):
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        os.environ["FLAGS_use_cuda"]       = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        paddle_device = f"gpu:{gpu_id}"
    else:
        os.environ["FLAGS_use_cuda"]       = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        paddle_device = "cpu"

    # ── Load model (once) ─────────────────────────────────────────────────────
    try:
        import paddle
        paddle.set_device(paddle_device)

        from paddleocr import TextRecognition
        rec = TextRecognition(model_name=model_name, device=paddle_device)
        print(json.dumps({"ready": True}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"model load failed: {exc}"}), flush=True)
        return

    # ── Request loop ──────────────────────────────────────────────────────────
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

        image_path = req.get("image")
        if not image_path:
            print(json.dumps({"error": "missing 'image' field"}), flush=True)
            continue

        try:
            out         = rec.predict(input=image_path, batch_size=1)
            text, score = _extract_text_score(out[0] if out else None)
            print(json.dumps({"text": text, "score": score}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
