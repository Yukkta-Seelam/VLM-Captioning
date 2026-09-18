#!/usr/bin/env python3
"""Run the RTSP YOLO26 detector, Insight output, and optional GenAI captions."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Full, Queue
import struct
import sys
import threading
import time
from urllib import error, request
from urllib.parse import urlparse

import cv2
import numpy as np
import pyneat
import yaml

T0 = time.monotonic()
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_SYSTEM_PROMPT = (
    "Describe the visible action in the detected person crop. Be factual, "
    "concise, and avoid guessing identity or protected attributes."
)
DEFAULT_USER_PROMPT = "What is the person doing in this crop?"

HISTORY_DIR = Path.home() / "detection_history"
HISTORY_IMAGES_DIR = HISTORY_DIR / "images"
HISTORY_LOG = HISTORY_DIR / "events.jsonl"
HISTORY_MAX_IMAGES = 300


def save_history_entry(image_rgb: np.ndarray, caption: str) -> None:
    from datetime import datetime, timezone

    HISTORY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    filename = ts.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    cv2.imwrite(str(HISTORY_IMAGES_DIR / filename), rgb_to_bgr(image_rgb))
    record = {"ts": ts.isoformat(), "caption": caption, "image": filename}
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    images = sorted(HISTORY_IMAGES_DIR.glob("*.jpg"))
    for stale in images[:-HISTORY_MAX_IMAGES]:
        stale.unlink(missing_ok=True)


@dataclass(frozen=True)
class Config:
    rtsp_url: str
    model_path: str
    labels_path: str
    frames: int
    min_score: float
    nms_iou: float
    max_detections: int
    classes: tuple[str, ...]
    timeout_ms: int
    debug: bool
    insight_host: str
    video_port: int
    metadata_port: int
    channel: int
    genai_enabled: bool
    genai_host: str
    genai_port: int
    genai_model: str
    genai_max_tokens: int
    genai_interval_seconds: float
    genai_timeout_seconds: float
    genai_max_pending_requests: int
    genai_system_prompt: str
    genai_user_prompt: str
    genai_trigger_port: int
    genai_min_sharpness: float
    genai_crop_margin: float
    genai_crop_top_margin: float
    genai_startup_grace_seconds: float
    genai_person_debounce_seconds: float
    genai_send_full_frame: bool
    genai_edge_margin_px: int
    genai_edge_check_enabled: bool
    genai_entry_settle_seconds: float


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = raw.get("source", {})
    model = raw.get("model", {})
    insight = raw.get("insight", {})
    inference = raw.get("inference", {})
    runtime = raw.get("runtime", {})
    genai_server = raw.get("genai_server", {})
    server_model = genai_server.get("model", {})
    genai = raw.get("genai", {})
    server_model_name = str(server_model.get("name", "") or "")
    genai_server_port = int(genai_server.get("port", 9998))
    return Config(
        rtsp_url=source.get("rtsp_url", ""),
        model_path=model.get("path", ""),
        labels_path=model.get("labels", ""),
        frames=int(inference.get("frames", 0)),
        min_score=float(inference.get("min_score", 0.55)),
        nms_iou=float(inference.get("nms_iou", 0.50)),
        max_detections=int(inference.get("max_detections", 24)),
        classes=class_filter(inference.get("classes")),
        timeout_ms=int(runtime.get("timeout_ms", 20000)),
        debug=bool(runtime.get("debug", False)),
        insight_host=insight.get("host", "127.0.0.1") or "127.0.0.1",
        video_port=int(insight.get("video_port", 9000)),
        metadata_port=int(insight.get("metadata_port", 9100)),
        channel=int(insight.get("channel", 0)),
        genai_enabled=bool(genai.get("enabled", False)),
        genai_host=genai.get("host", "127.0.0.1") or "127.0.0.1",
        genai_port=int(genai.get("port", genai_server_port)),
        genai_model=str(genai.get("model") or server_model_name),
        genai_max_tokens=int(genai.get("max_tokens", 128)),
        genai_interval_seconds=float(genai.get("interval_seconds", 5.0)),
        genai_timeout_seconds=float(genai.get("timeout_seconds", 30.0)),
        genai_max_pending_requests=max(1, int(genai.get("max_pending_requests", 1))),
        genai_system_prompt=genai.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        genai_user_prompt=genai.get("user_prompt", DEFAULT_USER_PROMPT),
        genai_trigger_port=int(genai.get("trigger_port", 9997)),
        genai_min_sharpness=float(genai.get("min_sharpness", 60.0)),
        genai_crop_margin=float(genai.get("crop_margin", 0.20)),
        genai_crop_top_margin=float(genai.get("crop_top_margin", 0.35)),
        genai_startup_grace_seconds=float(genai.get("startup_grace_seconds", 3.0)),
        genai_person_debounce_seconds=float(genai.get("person_debounce_seconds", 1.5)),
        genai_send_full_frame=bool(genai.get("send_full_frame", True)),
        genai_edge_margin_px=int(genai.get("edge_margin_px", 4)),
        genai_edge_check_enabled=bool(genai.get("edge_check_enabled", True)),
        genai_entry_settle_seconds=float(genai.get("entry_settle_seconds", 2.0)),
    )


def class_filter(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def load_labels(path: str) -> list[str]:
    label_path = Path(path)
    if not path or not label_path.is_file():
        return []
    return [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open RTSP source: {url}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0)) or 30
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP dimensions")
    return width, height, fps


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def decoded_tensor_to_rgb(tensor) -> np.ndarray:
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        stride = width
        try:
            strides = tensor.strides_bytes
            if strides:
                stride = int(strides[0])
        except Exception:
            pass

        # The accelerator often allocates decoded frames with row stride and/or
        # coded height padded past the display width/height (e.g. aligned to 64px).
        # Derive the true coded height from the payload size rather than assuming
        # it matches the display height, then crop each plane to width x height.
        total_rows = payload.size // stride
        coded_height = (total_rows * 2) // 3
        if coded_height < height:
            raise RuntimeError(
                f"NV12 payload too small: {payload.size} bytes, stride={stride}, "
                f"coded_height={coded_height} < height={height}"
            )

        y_plane = payload[: stride * coded_height].reshape(coded_height, stride)
        uv_start = stride * coded_height
        uv_rows = coded_height // 2
        uv_plane = payload[uv_start : uv_start + stride * uv_rows].reshape(uv_rows, stride)

        y_crop = y_plane[:height, :width]
        uv_crop = uv_plane[: height // 2, :width]
        nv12 = np.ascontiguousarray(np.vstack([y_crop, uv_crop]))

        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def is_tensor_like(value) -> bool:
    return hasattr(value, "copy_payload_bytes") and hasattr(value, "to_numpy")


def is_sample_like(value) -> bool:
    return hasattr(value, "kind") and hasattr(value, "fields")


def bbox_payload_from_tensors(tensors) -> bytes:
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return b""


def bbox_payload(result) -> bytes:
    if isinstance(result, (list, tuple)) and all(is_tensor_like(item) for item in result):
        return bbox_payload_from_tensors(result)

    if not is_sample_like(result):
        return b""

    stack = [result]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(current.fields)))
        if current.kind == pyneat.SampleKind.TensorSet:
            payload = bbox_payload_from_tensors(current.tensors)
            if payload:
                return payload
            continue
        if current.kind != pyneat.SampleKind.Tensor or current.tensor is None:
            continue
        fmt = (current.payload_tag or current.format or "").upper()
        if fmt and fmt != "BBOX":
            continue
        try:
            payload = current.tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return b""


def parse_boxes(result) -> list[dict]:
    payload = bbox_payload(result)
    if len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    boxes = []
    for idx in range(count):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, 4 + idx * 24)
        if w > 0 and h > 0:
            boxes.append(
                {
                    "bbox": [x, y, w, h],
                    "score": score,
                    "class_id": class_id,
                }
            )
    return boxes


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    """Return one branch of the combined output, by label or by combine order."""
    field = find_field(sample, label)
    if field is not None:
        return field
    fields = list(getattr(sample, "fields", []))
    if sample.kind == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"detector output missing {label} field")


def decoded_frame_tensor(sample):
    field = joined_field(sample, "frame", 0)
    if field.kind == pyneat.SampleKind.Tensor and field.tensor is not None:
        return field.tensor
    if field.kind == pyneat.SampleKind.TensorSet and field.tensors:
        return field.tensors[0]
    raise RuntimeError("detector output did not contain a decoded frame tensor")


def metadata_json(boxes: list[dict], labels: list[str], classes: tuple[str, ...] = ()) -> str:
    objects = []
    allowed = set(classes)
    for box in boxes:
        class_id = int(box["class_id"])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        if allowed and label.lower() not in allowed:
            continue
        objects.append(
            {
                "id": f"obj_{len(objects) + 1}",
                "label": label,
                "confidence": box["score"],
                "bbox": box["bbox"],
            }
        )
    return json.dumps({"objects": objects}, separators=(",", ":"))


def best_box_for_label(boxes: list[dict], labels: list[str], wanted: str):
    wanted = wanted.lower()
    matches = []
    for box in boxes:
        class_id = int(box["class_id"])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        if label.lower() == wanted:
            matches.append(box)
    return max(matches, key=lambda box: box["score"], default=None)


def crop_box(
    frame: np.ndarray,
    box: dict,
    margin: float = 0.0,
    top_margin: float | None = None,
) -> np.ndarray:
    """Crop the box out of frame, padded outward by `margin` (fraction of
    that side's own size) on the sides and bottom. `top_margin` overrides
    the top pad specifically (raw detection boxes tend to clip hair/heads
    tighter than the sides), defaulting to `margin` if not given. Padding is
    clamped to the actual frame bounds, so it silently shrinks near edges
    instead of failing."""
    x, y, w, h = [int(value) for value in box["bbox"]]
    height, width = frame.shape[:2]
    top_margin = margin if top_margin is None else top_margin

    pad_x = int(round(w * margin))
    pad_top = int(round(h * top_margin))
    pad_bottom = int(round(h * margin))

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_top)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_bottom)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid person crop: {box['bbox']}")
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def box_touches_edge(box: dict, width: int, height: int, margin_px: int = 4) -> bool:
    """True if the box's edge sits at (or past) the frame boundary. A person
    walking in through a doorway is a box clipped by the frame edge for as
    long as they're still entering/leaving -- unlike someone merely standing
    near the edge, whose box has real daylight on that side. `margin_px`
    absorbs box-coordinate quantization noise at a true boundary."""
    x, y, w, h = box["bbox"]
    return (
        x <= margin_px
        or y <= margin_px
        or x + w >= width - margin_px
        or y + h >= height - margin_px
    )


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def sharpness_score(crop_rgb: np.ndarray) -> float:
    """Laplacian-variance sharpness estimate. Higher = sharper; near-zero for
    a smeared/blurry or blown-out image with no real edges left."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def request_vlm_response(image_rgb: np.ndarray, cfg: Config) -> str:
    bgr = rgb_to_bgr(image_rgb)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("failed to encode image for the VLM")

    image = base64.b64encode(encoded.tobytes()).decode("ascii")
    payload = {
        "model": cfg.genai_model,
        "stream": True,
        "max_tokens": cfg.genai_max_tokens,
        "messages": [
            {"role": "system", "content": cfg.genai_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                    {
                        "type": "text",
                        "text": cfg.genai_user_prompt,
                    },
                ],
            },
        ],
    }

    url = f"http://{cfg.genai_host}:{cfg.genai_port}/v1/chat/completions"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_text = ""
    with request.urlopen(req, timeout=cfg.genai_timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
            response_text += delta.get("content") or ""
    return response_text.strip()


class GenAICommenter:
    def __init__(self, cfg: Config, labels: list[str], metadata_sender=None):
        self.cfg = cfg
        self.labels = labels
        self.metadata_sender = metadata_sender
        # Insight correlates metadata to the currently displayed frame by
        # timestamp, matched against the video's own RTP clock -- not wall
        # clock time. Track the most recent frame's pts so captions land on
        # (approximately) "now" in that same timeline instead of silently
        # never matching any frame.
        self.last_frame_pts_ns = -1
        self.queue: Queue[np.ndarray] = Queue(maxsize=cfg.genai_max_pending_requests)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.lock = threading.Lock()
        self.last_enqueue_at = 0.0
        self.in_flight = False
        self.server_available: bool | None = None
        self.response_count = 0
        self.started = False
        self.shutdown_requested = threading.Event()
        # Captions only happen when something calls request_caption() (e.g. a
        # button on the overlay page hitting /trigger), not on a timer. This
        # keeps video smooth continuously except for the brief moment a
        # caption is actually requested.
        self.manual_trigger = threading.Event()
        self.trigger_server: ThreadingHTTPServer | None = None
        # Auto-trigger on person count changes: someone entering (count goes
        # up) or the frame going empty (count drops to 0). A fresh process
        # (after each restart) has no memory of the prior frame, so the very
        # first observation just establishes a baseline instead of firing --
        # otherwise every restart would immediately "see" a still-present
        # person as if they had just walked in.
        self.last_person_count = 0
        self.person_state_initialized = False
        # RTSP reconnect plus model/decoder warm-up right after a (re)start
        # produce a few seconds of unstable detections (dropped frames,
        # low-confidence boxes) that can look like a real count change. Keep
        # tracking the baseline -- without scoring transitions -- until this
        # grace period has elapsed, so startup jitter can't read as someone
        # walking in and re-trigger a caption immediately after a restart.
        self.startup_grace_seconds = cfg.genai_startup_grace_seconds
        self.state_ready_at: float | None = None
        # A single missed/spurious detection (motion blur, brief occlusion)
        # looks identical to a real entry/exit if we react to one frame alone.
        # Require a candidate count to hold steady for a sustained duration
        # (not just N sampled frames, which can pass in well under a second
        # at this app's every-4th-frame sampling) before treating it as a
        # real transition.
        self.pending_person_count = 0
        self.pending_since: float | None = None
        self.person_debounce_seconds = cfg.genai_person_debounce_seconds
        # After an "entered" transition is confirmed, wait a bit longer
        # before actually capturing -- otherwise the crop catches someone
        # mid-walk-in, only partially in frame. No such delay for "left
        # frame": there's no one to crop until someone is detected again
        # anyway, so it just waits dormant for the next real entry.
        self.entry_settle_seconds = cfg.genai_entry_settle_seconds
        self.capture_ready_at: float | None = None
        # Avoids spamming the same "waiting for a sharper frame" line every
        # processed frame while we keep retrying.
        self.blur_wait_logged = False
        # Same idea for the "still crossing the frame edge" wait.
        self.edge_wait_logged = False

    def start(self) -> None:
        if self.cfg.genai_enabled and not self.started:
            self.worker.start()
            self._start_trigger_server()
            self.started = True

    def request_caption(self) -> None:
        self.manual_trigger.set()
        # Each new capture attempt gets its own shot at logging why it's
        # blocked -- without this, a block reason from a prior attempt (e.g.
        # a stale edge/blur wait) would permanently silence the message for
        # the rest of the process's life, even once a different attempt hits
        # the same wait for an unrelated reason.
        self.edge_wait_logged = False
        self.blur_wait_logged = False

    def _start_trigger_server(self) -> None:
        if not self.cfg.genai_trigger_port:
            return
        commenter = self

        class TriggerHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _reply(self, status: int, body: dict) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(payload)

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/trigger":
                    self._reply(404, {"error": "not found"})
                    return
                commenter.request_caption()
                self._reply(200, {"success": True})

        self.trigger_server = ThreadingHTTPServer(("0.0.0.0", self.cfg.genai_trigger_port), TriggerHandler)
        threading.Thread(target=self.trigger_server.serve_forever, daemon=True).start()
        print(f"[genai-server] trigger endpoint on http://0.0.0.0:{self.cfg.genai_trigger_port}/trigger", flush=True)

    def _count_persons(self, boxes: list[dict]) -> int:
        count = 0
        for box in boxes:
            class_id = int(box["class_id"])
            label = self.labels[class_id] if 0 <= class_id < len(self.labels) else ""
            if label.lower() == "person":
                count += 1
        return count

    def _check_person_transition(self, boxes: list[dict]) -> None:
        observed_count = self._count_persons(boxes)
        now = time.monotonic()

        if not self.person_state_initialized:
            self.last_person_count = observed_count
            self.person_state_initialized = True
            self.state_ready_at = now + self.startup_grace_seconds
            self.pending_person_count = observed_count
            self.pending_since = None
            return

        if now < self.state_ready_at:
            # Still inside the post-(re)start grace period: keep the
            # baseline tracking reality instead of scoring changes as
            # transitions.
            self.last_person_count = observed_count
            self.pending_person_count = observed_count
            self.pending_since = None
            return

        if observed_count == self.last_person_count:
            # Back to the confirmed count -- whatever candidate was building
            # up was just a blip, not a real change.
            self.pending_person_count = self.last_person_count
            self.pending_since = None
            return

        if observed_count != self.pending_person_count:
            self.pending_person_count = observed_count
            self.pending_since = now

        if self.pending_since is None or now - self.pending_since < self.person_debounce_seconds:
            return

        # The new count has held steady for long enough to count as real.
        person_count = self.pending_person_count
        if person_count > self.last_person_count:
            print(
                f"[auto-trigger] t={time.monotonic()-T0:.2f}s person count {self.last_person_count} "
                f"-> {person_count} (entered, capturing in {self.entry_settle_seconds:g}s)",
                flush=True,
            )
            self.capture_ready_at = time.monotonic() + self.entry_settle_seconds
        elif self.last_person_count > 0 and person_count == 0:
            print(f"[auto-trigger] person count {self.last_person_count} -> 0 (left frame)", flush=True)
            self.request_caption()
        self.last_person_count = person_count
        self.pending_since = None

    def try_enqueue(self, sample, boxes: list[dict]) -> None:
        if not self.cfg.genai_enabled:
            return
        self._check_person_transition(boxes)
        if self.capture_ready_at is not None:
            if time.monotonic() < self.capture_ready_at:
                return
            self.capture_ready_at = None
            self.request_caption()
        if not self.manual_trigger.is_set():
            return
        now = time.monotonic()
        if self._pending_count() >= self.cfg.genai_max_pending_requests:
            print("[genai-server] queue busy, dropping request", flush=True)
            return

        box = best_box_for_label(boxes, self.labels, "person")
        if box is None:
            # Nothing to capture right now (e.g. this was the "left frame"
            # trigger, which fires with no one on screen). Drop it instead of
            # leaving manual_trigger set -- otherwise it lingers until the
            # next person's very first visible frame and fires on them
            # immediately, skipping their entry debounce/settle wait entirely.
            self.manual_trigger.clear()
            return

        frame = decoded_tensor_to_rgb(decoded_frame_tensor(sample))
        # A box still touching the frame edge means the person is very likely
        # mid walk-in/walk-out, not actually settled -- wait for a frame where
        # they've cleared the boundary rather than capturing them half-cropped.
        if self.cfg.genai_edge_check_enabled and box_touches_edge(
            box, frame.shape[1], frame.shape[0], self.cfg.genai_edge_margin_px
        ):
            if not self.edge_wait_logged:
                print(
                    "[genai-server] person still crossing the frame edge, "
                    f"waiting for them to fully enter (bbox={box['bbox']} "
                    f"frame={frame.shape[1]}x{frame.shape[0]})",
                    flush=True,
                )
                self.edge_wait_logged = True
            return
        self.edge_wait_logged = False

        crop = crop_box(
            frame,
            box,
            margin=self.cfg.genai_crop_margin,
            top_margin=self.cfg.genai_crop_top_margin,
        )
        # Sharpness is always judged on the person crop, not the full frame --
        # a sharp background behind a motion-blurred subject shouldn't pass.
        sharpness = sharpness_score(crop)
        if sharpness < self.cfg.genai_min_sharpness:
            if not self.blur_wait_logged:
                print(
                    f"[genai-server] crop too blurry (sharpness={sharpness:.1f} < "
                    f"{self.cfg.genai_min_sharpness:g}), waiting for a sharper frame",
                    flush=True,
                )
                self.blur_wait_logged = True
            return
        self.blur_wait_logged = False

        # What's actually sent to the VLM: the full frame by default (avoids
        # a bad bbox clipping the person), or the tighter crop if configured
        # off. Either way the crop above still gates on the person's own
        # sharpness.
        image = frame if self.cfg.genai_send_full_frame else crop

        try:
            self.queue.put_nowait(image.copy())
            self.last_enqueue_at = now
            self.manual_trigger.clear()
            print(
                f"[timing] enqueued {'frame' if self.cfg.genai_send_full_frame else 'crop'} "
                f"t={time.monotonic()-T0:.2f}s sharpness={sharpness:.1f}",
                flush=True,
            )
        except Full:
            print("[genai-server] queue full, dropping request", flush=True)

    def close(self) -> None:
        self.stop_event.set()
        if self.trigger_server:
            self.trigger_server.shutdown()
        if self.started:
            self.worker.join(timeout=1.0)

    def _pending_count(self) -> int:
        with self.lock:
            return self.queue.qsize() + int(self.in_flight)

    def _set_in_flight(self, value: bool) -> None:
        with self.lock:
            self.in_flight = value

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                image = self.queue.get(timeout=0.2)
            except Empty:
                continue

            self._set_in_flight(True)
            try:
                if self._server_ready():
                    print(f"[timing] vlm request start t={time.monotonic()-T0:.2f}s", flush=True)
                    response = request_vlm_response(image, self.cfg)
                    print(f"[timing] vlm request done t={time.monotonic()-T0:.2f}s", flush=True)
                    if response:
                        self.response_count += 1
                        print(
                            f"\n[response #{self.response_count:03d}] {response}",
                            flush=True,
                        )
                        # The MLA accelerator is shared with genai_server.py and does not
                        # reliably release after a GenAI request while this process keeps
                        # running; restart proactively so the supervisor loop gets a fresh
                        # pipeline quickly instead of waiting for the stall-detection timeout.
                        self.shutdown_requested.set()
                        try:
                            save_history_entry(image, response)
                        except Exception as exc:
                            print(f"[history] failed to save entry: {exc}", flush=True)
                        if self.metadata_sender is not None:
                            try:
                                # Deliberately -1 (no timestamp): Insight's viewer stores
                                # timestamped metadata in an exact-match map keyed to a
                                # specific video frame, which a caption arriving several
                                # seconds after its source frame will never match again.
                                # An untimestamped message instead goes into an "arrival"
                                # fallback queue that the very next rendered frame picks
                                # up regardless of its own timestamp -- what we actually
                                # want for "show this now" rather than "show this exact
                                # frame's caption".
                                ok = self.metadata_sender.send_metadata(
                                    "caption",
                                    json.dumps({
                                        "text": response,
                                        # Distinguishes this caption from the previous one
                                        # even if the text happens to repeat, so the viewer
                                        # can tell "new caption" apart from "same caption,
                                        # already dismissed" -- response_count resets to 1
                                        # on every restart (we restart after each caption),
                                        # so wall-clock ms is what's actually unique here.
                                        "id": int(time.time() * 1000),
                                    }),
                                    -1,
                                    "",
                                )
                                if not ok:
                                    print("[genai-server] failed to send caption to Insight", flush=True)
                            except Exception as exc:
                                print(f"[genai-server] failed to send caption to Insight: {exc}", flush=True)
            except (TimeoutError, OSError, error.URLError) as exc:
                print(f"[genai-server] request failed: {exc}", flush=True)
            except Exception as exc:
                print(f"[genai-server] request failed: {exc}", flush=True)
            finally:
                self._set_in_flight(False)
                self.queue.task_done()

    def _server_ready(self) -> bool:
        url = f"http://{self.cfg.genai_host}:{self.cfg.genai_port}/v1/models"
        try:
            timeout = min(self.cfg.genai_timeout_seconds, 5.0)
            with request.urlopen(url, timeout=timeout) as res:
                ready = 200 <= res.status < 300
                if ready and self.server_available is False:
                    print(
                        f"\n[genai-server] connected "
                        f"http://{self.cfg.genai_host}:{self.cfg.genai_port}",
                        flush=True,
                    )
                self.server_available = ready
                return ready
        except (TimeoutError, OSError, error.URLError) as exc:
            if self.server_available is not False:
                print(
                    f"[genai-server] waiting for "
                    f"http://{self.cfg.genai_host}:{self.cfg.genai_port}: {exc}",
                    flush=True,
                )
            self.server_available = False
            return False


def build_source_graph(cfg: Config, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.payload_type = 96
    opt.insert_queue = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.sima_allocator_type = 2
    opt.decoder_raw_output = True
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return pyneat.groups.rtsp_decoded_input(opt)


def build_video_graph(cfg: Config, width: int, height: int, fps: int):
    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight_host
    sender_opt.channel = cfg.channel
    sender_opt.video_port_base = cfg.video_port

    graph = pyneat.Graph("video")
    graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender_opt))
    return graph


def build_model(cfg: Config, width: int, height: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.input_max_width = width
    opt.preprocess.input_max_height = height
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def build_pipeline(cfg: Config, width: int, height: int, fps: int):
    model = build_model(cfg, width, height)
    source = build_source_graph(cfg, width, height, fps)
    video_graph = build_video_graph(cfg, width, height, fps)

    # Insight correlates the RTP timestamp with the metadata timestamp, so the encoder and the
    # detections must stay in one Run and therefore on one GStreamer timeline. The frame branch
    # returns the decoded frame the GenAI commenter crops.
    branch = pyneat.graphs.branch("source", ["video", "model", "frame"])

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))

    joined = pyneat.graphs.combine(
        ["frame", "detections"], "detector_output", pyneat.CombinePolicy.ByFrame
    )

    graph = pyneat.Graph()
    graph.connect(source, branch)
    graph.connect(branch, video_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, detections_graph)
    graph.connect(branch, frame_graph)
    graph.connect(frame_graph, joined)
    graph.connect(detections_graph, joined)

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return model, graph, graph.build(run_opt)


def build_metadata_sender(cfg: Config):
    opt = pyneat.MetadataSenderOptions()
    opt.host = cfg.insight_host
    opt.channel = cfg.channel
    opt.metadata_port_base = cfg.metadata_port
    return pyneat.MetadataSender(opt)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detection-to-VLM assistant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if not cfg.rtsp_url or not cfg.model_path:
        print("config requires source.rtsp_url and model.path", file=sys.stderr)
        return 2
    if not Path(cfg.model_path).is_file():
        print(f"model package does not exist: {cfg.model_path}", file=sys.stderr)
        return 2
    if cfg.genai_enabled and not cfg.genai_model:
        print("config requires genai.model or genai_server.model.name", file=sys.stderr)
        return 2

    detector_run = commenter = None
    t0 = time.monotonic()
    try:
        labels = load_labels(cfg.labels_path)
        width, height, fps = probe_rtsp(cfg.rtsp_url)
        print(f"[timing] probe_rtsp done t={time.monotonic()-t0:.2f}s", flush=True)
        _model, _detector_graph, detector_run = build_pipeline(cfg, width, height, fps)
        print(f"[timing] pipeline built t={time.monotonic()-t0:.2f}s", flush=True)
        metadata = build_metadata_sender(cfg)
        commenter = GenAICommenter(cfg, labels, metadata_sender=metadata)
        commenter.start()
        print(
            f"[detector] stream {cfg.rtsp_url}\n"
            f"[detector] input {width}x{height}@{fps}\n"
            f"[insight] video={cfg.insight_host}:{cfg.video_port} "
            f"metadata={cfg.insight_host}:{cfg.metadata_port} channel={cfg.channel}\n"
            f"[genai-server] "
            f"{'enabled' if cfg.genai_enabled else 'disabled'} "
            f"model={cfg.genai_model or '-'} "
            f"url=http://{cfg.genai_host}:{cfg.genai_port} "
            f"trigger-only (POST /trigger on :{cfg.genai_trigger_port})\n"
        )

        processed = 0
        while cfg.frames <= 0 or processed < cfg.frames:
            if commenter.shutdown_requested.is_set():
                print("restarting after caption to avoid MLA contention", file=sys.stderr)
                break
            sample = detector_run.pull("detector_output", cfg.timeout_ms)
            if sample is None:
                print("RTSP stream ended or pull timed out", file=sys.stderr)
                break
            if processed == 0:
                print(f"[timing] first frame t={time.monotonic()-t0:.2f}s", flush=True)
            boxes = parse_boxes(joined_field(sample, "detections", 1))
            commenter.last_frame_pts_ns = sample.pts_ns
            commenter.try_enqueue(sample, boxes)
            ok = metadata.send_metadata(
                "object-detection",
                metadata_json(boxes, labels, cfg.classes),
                int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1,
                str(sample.frame_id) if sample.frame_id >= 0 else "",
            )
            if not ok:
                raise RuntimeError("failed to send metadata to Insight")
            processed += 1
            if cfg.debug:
                print(f"[detector] frame={processed} detections={len(boxes)}")
        return 0 if processed > 0 else 3
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if commenter is not None:
            commenter.close()
        if detector_run is not None:
            detector_run.close()


if __name__ == "__main__":
    raise SystemExit(main())
