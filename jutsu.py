import cv2
import mediapipe as mp
import numpy as np
import math
import os

# 🌀 SHADOW CLONE JUTSU - CLEAN VERSION
# Optimized for Live Shadow Clones without frame-lag or background subtraction

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the Hand Detector and Segmenter
model_paths = {
    'hands': 'hand_landmarker.task',
    'selfie': 'selfie_segmenter.task'
}

# Hand Landmarker
base_options_hands = python.BaseOptions(model_asset_path=model_paths['hands'])
options_hands = vision.HandLandmarkerOptions(
    base_options=base_options_hands,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options_hands)

# Image Segmenter (Selfie)
base_options_selfie = python.BaseOptions(model_asset_path=model_paths['selfie'])
options_selfie = vision.ImageSegmenterOptions(
    base_options=base_options_selfie,
    running_mode=vision.RunningMode.IMAGE,
    output_category_mask=True # More reliable than confidence mask across OS
)
segmenter = vision.ImageSegmenter.create_from_options(options_selfie)

def detect_shadow_clone_sign(hand_landmarks_list):
    """Detects if both hands are visible to trigger the jutsu."""
    if len(hand_landmarks_list) < 2:
        return False, "SHOW BOTH HANDS"
    return True, "KAGE BUNSHIN NO JUTSU!"

def generate_clone_positions(fw, fh, num_clones=12):
    """
    Tight Army Layout (Crowd Style).
    Positions 12 clones tightly packed behind the user.
    """
    clones = [
        # ROW 3 (Back)
        (int(fw * -0.38), int(fh * -0.05), 0.65),
        (int(fw * 0.38), int(fh * -0.05), 0.65),
        (int(fw * -0.24), int(fh * -0.08), 0.70),
        (int(fw * 0.24), int(fh * -0.08), 0.70),
        # ROW 2 (Middle)
        (int(fw * -0.30), int(fh * -0.02), 0.80),
        (int(fw * 0.30), int(fh * -0.02), 0.80),
        (int(fw * -0.16), int(fh * -0.04), 0.85),
        (int(fw * 0.16), int(fh * -0.04), 0.85),
        # ROW 1 (Front - Near Shoulders)
        (int(fw * -0.20), 0, 0.90),
        (int(fw * 0.20), 0, 0.90),
        (int(fw * -0.10), int(fh * 0.02), 0.95),
        (int(fw * 0.10), int(fh * 0.02), 0.95),
    ]
    return clones[:num_clones]


def overlay_guide_icon(frame, icon, is_active):
    """Overlays the hand sign guide at the bottom center."""
    if icon is None: return
    fh, fw = frame.shape[:2]
    icon_size = 120
    resized = cv2.resize(icon, (icon_size, icon_size))
    ih, iw = resized.shape[:2]
    x_pos, y_pos = (fw - iw) // 2, fh - ih - 25
    
    # Background Box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_pos-8, y_pos-30), (x_pos+iw+8, y_pos+ih+8), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Border & Label
    color = (0, 255, 0) if is_active else (0, 200, 255)
    cv2.rectangle(frame, (x_pos-8, y_pos-30), (x_pos+iw+8, y_pos+ih+8), color, 2)
    label = "JUTSU ACTIVATED!" if is_active else "PERFORM THIS SIGN"
    cv2.putText(frame, label, (x_pos + (iw-150)//2, y_pos-8), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    # Alpha Blend Icon
    if resized.shape[2] == 4:
        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            frame[y_pos:y_pos+ih, x_pos:x_pos+iw, c] = alpha * resized[:, :, c] + (1 - alpha) * frame[y_pos:y_pos+ih, x_pos:x_pos+iw, c]
    else:
        frame[y_pos:y_pos+ih, x_pos:x_pos+iw] = resized

# Global State
jutsu_active = False
clone_positions = []
clone_fade_in = 0
jutsu_persistence = 0  # To prevent blinking/flickering

def get_feathered_mask(w, h):
    """Creates a sharper mask with less blur to fix the 'too blur' issue."""
    mask = np.zeros((h, w), dtype=np.float32)
    # Slightly larger rectangle, smaller blur for sharper look
    cv2.rectangle(mask, (int(w*0.05), int(h*0.02)), (int(w*0.95), int(h*0.98)), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (int(w*0.1)|1, int(h*0.1)|1), 0)
    return mask

def check_sign(hand_landmarks):
    """
    Simpler check: As long as Index or Middle is straight-ish.
    This makes it much easier to trigger the jutsu.
    """
    def dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]
    index_pip = hand_landmarks[6]
    middle_tip = hand_landmarks[12]
    middle_pip = hand_landmarks[10]
    
    # Extended if tip is further from wrist than PIP
    index_ext = dist(index_tip, wrist) > dist(index_pip, wrist) * 1.1
    middle_ext = dist(middle_tip, wrist) > dist(middle_pip, wrist) * 1.1
    
    return index_ext or middle_ext # Either index OR middle is enough

# Load Guide
guide_icon = cv2.imread(os.path.join('image', 'hand.png'), cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("--- SHADOW CLONE JUTSU READY ---")

mask_buffer = None

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    # Mediapipe Conversion
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # --- LOOP OPTIMIZATION ---
    # 1. Detect Hands First (Fastest)
    result = None
    try:
        result = detector.detect(mp_image)
    except: pass
    
    detect_count = len(result.hand_landmarks) if (result and result.hand_landmarks) else 0
    jutsu_detected_this_frame = False
    
    if detect_count >= 2:
        # Relaxed check: as long as fingers are visible
        valid_hands = 0
        for hand in result.hand_landmarks:
            if check_sign(hand):
                valid_hands += 1
        if valid_hands >= 2:
            jutsu_detected_this_frame = True

    # 2. Run Heavy Segmenter ONLY if needed
    selfie_mask = None
    if jutsu_active or jutsu_persistence > 0 or jutsu_detected_this_frame:
        try:
            segmentation_result = segmenter.segment(mp_image)
            # Use numpy_view() and squeeze to ensure shape is (H, W) and not (H, W, 1)
            mask_raw = np.squeeze(segmentation_result.category_mask.numpy_view())
            selfie_mask = (mask_raw > 0).astype(np.float32) # Normalized to 0.0 - 1.0
        except Exception as e:
            print(f"[RECOVERABLE] AI Segmenter error: {e}")

    # 3. Render Shadow Clone Army
    if (jutsu_active or jutsu_persistence > 0) and selfie_mask is not None:
        # Find the bounding box of the user in the mask
        y_locs, x_locs = np.where(selfie_mask > 0.1)
        if len(y_locs) > 100 and len(x_locs) > 100: # Ensure valid person detected
            min_y, max_y = np.min(y_locs), np.max(y_locs)
            min_x, max_x = np.min(x_locs), np.max(x_locs)
            
            user_cx = (min_x + max_x) // 2
            user_cy = (min_y + max_y) // 2
            
            # Mask out the user exactly
            condition = np.stack((selfie_mask,) * 3, axis=-1)
            masked_frame = np.where(condition > 0.1, frame, 0).astype(np.uint8)
            
            # Crop exactly to the user's size (no empty space!)
            user_crop = masked_frame[min_y:max_y, min_x:max_x]
            mask_crop = selfie_mask[min_y:max_y, min_x:max_x]
            
            # Draw Clones (Back to Front)
            for (x_offset, y_offset, scale) in clone_positions:
                if clone_fade_in < 0.05: continue
                
                # Resize only the crop!
                cw = int((max_x - min_x) * scale)
                ch = int((max_y - min_y) * scale)
                if cw <= 0 or ch <= 0: continue
                
                # Position clone relative to the actual user
                cx = user_cx - cw // 2 + x_offset
                cy = user_cy - ch // 2 + y_offset
                
                c_crop = cv2.resize(user_crop, (cw, ch))
                m_crop = cv2.resize(mask_crop, (cw, ch))
                
                # Depth Blur
                blur_val = int((1.0 - scale) * 15) | 1
                if blur_val > 1:
                    c_crop = cv2.GaussianBlur(c_crop, (blur_val, blur_val), 0)
                
                # Alpha Blending Bounds
                dx1, dy1 = max(0, cx), max(0, cy)
                dx2, dy2 = min(fw, cx + cw), min(fh, cy + ch)
                s_x1, s_y1 = max(0, dx1 - cx), max(0, dy1 - cy)
                s_x2, s_y2 = cw - max(0, (cx + cw) - fw), ch - max(0, (cy + ch) - fh)
                
                if dx2 > dx1 and dy2 > dy1:
                    alpha = np.stack([m_crop[s_y1:s_y2, s_x1:s_x2]] * 3, axis=-1) * clone_fade_in
                    blended = (c_crop[s_y1:s_y2, s_x1:s_x2].astype(np.float32) * alpha + 
                               frame[dy1:dy2, dx1:dx2].astype(np.float32) * (1.0 - alpha))
                    frame[dy1:dy2, dx1:dx2] = blended.astype(np.uint8)
            
            # 4. Draw ORIGINAL YOU back on top natively (Removes ghosting/whiteout)
            main_mask = np.stack((mask_crop > 0.3,) * 3, axis=-1)
            frame[min_y:max_y, min_x:max_x] = np.where(
                main_mask,
                user_crop,
                frame[min_y:max_y, min_x:max_x]
            )

    # 4. Handle State
    status_text = "SHOW BOTH HANDS"
    status_color = (0, 255, 0)
    
    if jutsu_detected_this_frame:
        if not jutsu_active:
            print("[JUTSU] ACTIVATED! Shadow Clones incoming!")
            clone_positions = generate_clone_positions(fw, fh)
            clone_fade_in = 0
        jutsu_active = True
        jutsu_persistence = 40 
        status_text = "KAGE BUNSHIN NO JUTSU!"
        status_color = (0, 255, 255)
        cv2.rectangle(frame, (5, 5), (fw-5, fh-5), (0, 200, 255), 4)
        clone_fade_in = min(1.0, clone_fade_in + 0.15)
    else:
        if jutsu_persistence > 0:
            jutsu_persistence -= 1
        else:
            if jutsu_active:
                print("[JUTSU] Released.")
                clone_fade_in = max(0, clone_fade_in - 0.1)
                if clone_fade_in <= 0: jutsu_active = False
            status_color = (0, 165, 255)

    # 5. UI & Debug Info
    detect_count = len(result.hand_landmarks) if (result and result.hand_landmarks) else 0
    if detect_count > 0:
        print(f"[AI] Hand Count: {detect_count}/2", end='\r')
    if result and result.hand_landmarks:
        for hand in result.hand_landmarks:
            for conn in [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (5,9), (9,13), (13,17), (0,17), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20)]:
                p1, p2 = hand[conn[0]], hand[conn[1]]
                cv2.line(frame, (int(p1.x * fw), int(p1.y * fh)), (int(p2.x * fw), int(p2.y * fh)), (0, 255, 0), 2)
            for lm in hand:
                cv2.circle(frame, (int(lm.x * fw), int(lm.y * fh)), 5, (0, 0, 255), -1)

    overlay_guide_icon(frame, guide_icon, jutsu_active)
    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, status_color, 2)
    cv2.putText(frame, f"Hands: {detect_count}/2", (30, 90), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    
    cv2.imshow("Kage Bunshin no Jutsu", frame)
    try: cv2.setWindowProperty("Kage Bunshin no Jutsu", cv2.WND_PROP_TOPMOST, 1) # Keep on top
    except: pass

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("--- SHUTDOWN ---")
