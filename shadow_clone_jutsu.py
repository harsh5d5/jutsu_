"""
🍥 NARUTO SHADOW CLONE JUTSU — Person-Only Edition
===================================================
Requirements:
    pip install mediapipe opencv-python numpy

Models needed (place in SAME FOLDER as this script):
    hand_landmarker.task
        → https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    selfie_segmenter.tflite
        → https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite

Keys:  Q = quit   |   R = reset clones
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import sys

# ── Model paths (same folder as script) ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
SEG_MODEL  = os.path.join(SCRIPT_DIR, "selfie_segmenter.tflite")

# ── Task API imports ─────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
RunningMode = mp.tasks.vision.RunningMode


# =============================================================================
# GESTURE: Clone Seal Detection
# =============================================================================
def dist3d(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def is_clone_seal(hands):
    if len(hands) < 2:
        return False
    h1, h2 = hands[0], hands[1]

    def up(lms, tip, pip):
        return lms[tip].y < lms[pip].y

    idx_ok  = up(h1, 8, 6) and up(h2, 8, 6)
    curl_ok = (not up(h1, 12, 10)) or (not up(h1, 16, 14))
    near_ok = dist3d(h1[0], h2[0]) < 0.42
    return idx_ok and curl_ok and near_ok


# =============================================================================
# PERSON EXTRACTION — strips background, returns person-only image + mask
# =============================================================================
def extract_person(frame, raw_mask):
    """Convert raw segmentation confidence into a clean person cutout."""
    hard = (raw_mask > 0.60).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hard = cv2.erode(hard, kernel, iterations=1)
    soft = cv2.GaussianBlur(hard.astype(np.float32), (11, 11), 0)
    person_img = (frame.astype(np.float32) * soft[:, :, None]).astype(np.uint8)
    return person_img, soft


# =============================================================================
# CLONE RENDER — draws a person-only clone (no background bleed)
# =============================================================================
def draw_clone(canvas, person_img, person_mask, dx, dy, scale, fade=1.0):
    h, w = canvas.shape[:2]
    M = np.float32([
        [scale, 0,     dx + (1 - scale) * w / 2],
        [0,     scale, dy + (1 - scale) * h / 2],
    ])
    w_img  = cv2.warpAffine(person_img, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_mask = cv2.warpAffine(person_mask, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Slight blue tint → looks like a shadow clone
    tinted = w_img.astype(np.float32)
    tinted[:, :, 0] *= 1.05   # blue channel boost
    tinted[:, :, 1] *= 0.92   # green slightly down
    tinted[:, :, 2] *= 0.90   # red slightly down
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    m3 = (w_mask * fade)[:, :, None]
    canvas[:] = (tinted * m3 + canvas * (1 - m3)).astype(np.uint8)


# =============================================================================
# SMOKE POOF
# =============================================================================
class Poof:
    DUR = 0.4

    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy
        self.t0 = time.time()

    @property
    def alive(self):
        return (time.time() - self.t0) < self.DUR

    def draw(self, canvas):
        h, w = canvas.shape[:2]
        age   = time.time() - self.t0
        alpha = (1 - age / self.DUR) * 0.75
        r     = max(1, int(h * 0.14 * (0.4 + age * 3)))
        ov    = canvas.copy()
        cv2.circle(ov, (w // 2 + self.dx, h // 2 + self.dy), r, (252, 254, 255), -1)
        cv2.addWeighted(ov, alpha, canvas, 1 - alpha, 0, canvas)


# =============================================================================
# CLONE DATA — positions & fade-in
# =============================================================================
class Clone:
    SLOTS = [
        (-0.20,  0.04, 0.97),   # Left side
        ( 0.20,  0.04, 0.97),   # Right side
        (-0.38, -0.02, 0.90),   # Far left back
        ( 0.38, -0.02, 0.90),   # Far right back
        ( 0.00, -0.08, 0.83),   # Center deep back
    ]

    def __init__(self, idx, w, h):
        rx, ry, s = self.SLOTS[idx % len(self.SLOTS)]
        self.dx    = int(rx * w)
        self.dy    = int(ry * h)
        self.scale = s
        self.t0    = time.time()

    @property
    def fade(self):
        return min((time.time() - self.t0) / 0.35, 1.0)


# =============================================================================
# SHADOW CLONE MANAGER
# =============================================================================
class ShadowCloneManager:
    SPAWN_INTERVAL = 0.55
    MAX_CLONES     = 5

    def __init__(self):
        self._verify_models()

        # Hand landmarker
        self.hand_det = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            ))
        print("✅ Hand landmarker loaded")

        # Selfie segmenter
        self.segmenter = ImageSegmenter.create_from_options(
            ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=SEG_MODEL),
                running_mode=RunningMode.VIDEO,
                output_confidence_masks=True,
            ))
        print("✅ Segmenter loaded")

        self.clones     = []
        self.poofs      = []
        self.seal_start = None
        self.last_spawn = 0
        self._snap_img  = None
        self._snap_mask = None

    def _verify_models(self):
        missing = []
        if not os.path.isfile(HAND_MODEL):
            missing.append(f"  ✗ {HAND_MODEL}")
        if not os.path.isfile(SEG_MODEL):
            missing.append(f"  ✗ {SEG_MODEL}")
        if missing:
            print("\n❌ Missing model files:\n" + "\n".join(missing))
            print("\nDownload links:")
            print("  Hand: https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            print("  Seg:  https://storage.googleapis.com/mediapipe-models/"
                  "image_segmenter/selfie_segmenter/float16/latest/"
                  "selfie_segmenter.tflite")
            print(f"\nSave them to: {SCRIPT_DIR}\n")
            sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    def process(self, frame, ts_ms):
        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ── Segmentation (person mask) ───────────────────────────────────────
        seg_res  = self.segmenter.segment_for_video(mp_img, ts_ms)
        raw_mask = seg_res.confidence_masks[0].numpy_view()
        p_img, p_mask = extract_person(frame, raw_mask)

        # Keep a snapshot of the person (only when person is visible)
        if p_mask.max() > 0.25:
            self._snap_img  = p_img.copy()
            self._snap_mask = p_mask.copy()

        # ── Hand detection ───────────────────────────────────────────────────
        hands = self.hand_det.detect_for_video(mp_img, ts_ms).hand_landmarks or []
        seal  = is_clone_seal(hands)

        # ── Spawn clones ─────────────────────────────────────────────────────
        now = time.time()
        if seal:
            if not self.seal_start:
                self.seal_start = now
                self.last_spawn = now - 0.1
            if (len(self.clones) < self.MAX_CLONES
                    and now - self.last_spawn >= self.SPAWN_INTERVAL):
                c = Clone(len(self.clones), w, h)
                self.clones.append(c)
                self.poofs.append(Poof(c.dx, c.dy))
                self.last_spawn = now
        else:
            self.seal_start = None

        # ── Render ───────────────────────────────────────────────────────────
        output = frame.copy()

        if self._snap_img is not None:
            # Draw clones sorted by dy (furthest first = behind)
            for c in sorted(self.clones, key=lambda x: x.dy):
                draw_clone(output, self._snap_img, self._snap_mask,
                           c.dx, c.dy, c.scale, fade=c.fade)

            # Real user ALWAYS on top — paste live person over everything
            m3 = p_mask[:, :, None]
            output[:] = (frame * m3 + output * (1 - m3)).astype(np.uint8)

        # ── Smoke effects ────────────────────────────────────────────────────
        self.poofs = [p for p in self.poofs if p.alive]
        for p in self.poofs:
            p.draw(output)

        # ── HUD ──────────────────────────────────────────────────────────────
        color = (0, 255, 120) if seal else (0, 160, 255)
        label = (f"SEAL ACTIVE  |  Clones: {len(self.clones)}/{self.MAX_CLONES}"
                 if seal
                 else f"Clones: {len(self.clones)}  |  Show Clone Seal!")
        # Drop shadow for readability
        cv2.putText(output, label, (22, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, label, (20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, color, 2, cv2.LINE_AA)

        if seal and len(self.clones) == self.MAX_CLONES:
            cv2.putText(output, "SHADOW CLONE JUTSU!",
                        (w // 2 - 175, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 1.05, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(output, "SHADOW CLONE JUTSU!",
                        (w // 2 - 175, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 1.05, (0, 220, 255), 2, cv2.LINE_AA)

        return output


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    print("=" * 55)
    print("  🍥  Shadow Clone Jutsu  — Person-Only Edition")
    print("=" * 55)

    manager = ShadowCloneManager()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("\nCamera ready!  Q = quit   R = reset clones\n")

    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int((time.time() - t0) * 1000)
        cv2.imshow("Shadow Clone Jutsu  [Q=quit  R=reset]",
                   manager.process(frame, ts_ms))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            manager.clones = []
            print("Clones dismissed!")

    cap.release()
    cv2.destroyAllWindows()
    print("Jutsu released. 🌀")


if __name__ == "__main__":
    main()
