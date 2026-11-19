"""Hand detection for follow_object --hand: a visible hand (open or closed) is followed, no hand sends
the robot home, and with --gripper the claw copies the hand (fist = closed, open hand = open).

MediaPipe's gesture recognizer (gesture_recognizer.task, fetched by scripts/download_models.sh) gives the
21 hand landmarks plus a trained gesture label. The open/closed call combines that label with a simple
count of extended fingers from the landmarks, so one wrong label does not move the claw. HandGate then
requires each call to hold for a moment before the robot starts following, goes home, or moves the claw.
"""
import os
import threading
import time

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gesture_recognizer.task')

WRIST = 0
PALM = [0, 5, 9, 13, 17]                 # wrist + the four finger bases: the palm centre is their mean
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]   # (PIP joint, tip) of index, middle, ring, little
EXTENDED_RATIO = 1.1                     # tip this much farther from the wrist than the PIP joint = straight
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [0, 17, 18, 19, 20],
          [5, 9, 13, 17]]                # for drawing

PRESENT_CONFIRM = 0.3                    # s a hand must be visible before the robot follows it
HOME_DELAY = 0.5                         # s without a hand before the robot goes home
GRIP_CONFIRM = 0.3                       # s a fist (open hand) must last before the claw closes (opens)


def fingers_extended(landmarks):
    """How many of index, middle, ring and little fingers are straight. `landmarks`: (21, 2) pixels.

    A straight finger's tip is farther from the wrist than its middle (PIP) joint; a curled one folds
    back toward the palm. Distances to the wrist do not depend on how the hand is turned in the image.
    The thumb is left out: it is unreliable from most viewpoints.
    """
    p = np.asarray(landmarks, float)[:, :2]
    wrist = p[WRIST]
    return sum(np.linalg.norm(p[tip] - wrist) > EXTENDED_RATIO * np.linalg.norm(p[pip] - wrist)
               for pip, tip in FINGERS)


def classify(gesture, n_extended):
    """'open', 'closed' or 'uncertain' from the recognizer's label and the finger count."""
    if gesture == 'Open_Palm' or n_extended >= 4:
        return 'open'
    if gesture == 'Closed_Fist' or n_extended == 0:
        return 'closed'
    return 'uncertain'


class Hand:
    """One detected hand, in pixels."""

    def __init__(self, landmarks, gesture, score):
        self.landmarks = np.asarray(landmarks, float)        # (21, 2)
        self.gesture, self.score = gesture, score
        self.fingers = fingers_extended(self.landmarks)
        self.state = classify(gesture, self.fingers)
        self.center = self.landmarks[PALM].mean(axis=0)

    def as_detection(self):
        """Same tuple as detection.detect(): (cx, cy, area, contour, shape distance)."""
        hull = cv2.convexHull(self.landmarks.astype(np.float32))
        return float(self.center[0]), float(self.center[1]), float(cv2.contourArea(hull)), None, None


class HandDetector:
    """MediaPipe gesture recognizer in video mode (it tracks the hand between frames)."""

    def __init__(self, model_path=MODEL_PATH, min_confidence=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{os.path.relpath(model_path)} missing: run scripts/download_models.sh')
        os.environ.setdefault('GLOG_minloglevel', '2')      # MediaPipe's start-up log is very chatty
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence)
        self._rec = vision.GestureRecognizer.create_from_options(options)
        self._last_ms = -1

    def detect(self, bgr):
        """The hand in a BGR frame, or None."""
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ms = max(self._last_ms + 1, int(time.perf_counter() * 1000))   # video mode: strictly increasing
        self._last_ms = ms
        result = self._rec.recognize_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.hand_landmarks:
            return None
        landmarks = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
        top = result.gestures[0][0] if result.gestures and result.gestures[0] else None
        return Hand(landmarks, top.category_name if top else None, top.score if top else 0.0)

    def close(self):
        self._rec.close()


class AsyncHandDetector:
    """Runs a HandDetector in a background thread on the newest submitted frame.

    The recognizer takes ~30 ms per frame on a CPU, more than the 20 ms servo period: called inline it
    would slow servoL down and make the motion jerky. Frames that arrive while it is busy are skipped.
    """

    def __init__(self, detector):
        self._det = detector
        self._lock, self._wake, self._stop = threading.Lock(), threading.Event(), threading.Event()
        self._frame, self._result = None, (0, None)
        self._thread = threading.Thread(target=self._run, name='hand-detector', daemon=True)
        self._thread.start()

    def submit(self, bgr):
        with self._lock:
            self._frame = bgr.copy()     # the camera may reuse its buffer
        self._wake.set()

    def latest(self):
        """(count, hand): `count` goes up by one per processed frame, `hand` is a Hand or None."""
        with self._lock:
            return self._result

    def _run(self):
        while not self._stop.is_set():
            if not self._wake.wait(0.1):
                continue
            with self._lock:
                frame, self._frame = self._frame, None
                self._wake.clear()
            if frame is None:
                continue
            try:
                hand = self._det.detect(frame)
            except Exception as e:       # keep the thread alive; the robot sees "no hand" and goes home
                print(f'Hand detection failed: {e}', flush=True)
                hand = None
            with self._lock:
                self._result = (self._result[0] + 1, hand)

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
        self._det.close()


class HandGate:
    """Per-frame hand calls -> a stable mode ('follow' or 'home') and claw state ('open' / 'closed').

    `mode`: the robot follows any hand it sees, open or closed. Follow once a hand has been visible for
    PRESENT_CONFIRM; go home once no hand has been seen for HOME_DELAY. Starts in 'home'.

    `grip`: 'closed' once a fist has been SEEN for GRIP_CONFIRM, 'open' once an open hand has, None
    before either. Losing the hand does not change it, so the claw never drops or grabs something
    because the hand left the image. 'uncertain' frames (pointing, two fingers) neither move the claw
    nor interrupt a fist or open hand that is being confirmed.
    """

    def __init__(self, present_confirm=PRESENT_CONFIRM, home_delay=HOME_DELAY, grip_confirm=GRIP_CONFIRM):
        self.present_confirm, self.home_delay, self.grip_confirm = present_confirm, home_delay, grip_confirm
        self.mode = 'home'
        self.grip = None
        self.present_since = None        # first frame of the current run of visible-hand frames
        self.last_seen = -1e9
        self.run = None                  # (clear call, since): consecutive seen fist / open hand

    def update(self, state, now):
        """`state`: 'open', 'closed', 'uncertain', or None for no hand."""
        if state is None:
            self.present_since, self.run = None, None    # a fist must be seen again from the start
            if self.mode == 'follow' and now - self.last_seen >= self.home_delay:
                self.mode = 'home'
            return self.mode
        self.last_seen = now
        if self.present_since is None:
            self.present_since = now
        if self.mode == 'home' and now - self.present_since >= self.present_confirm:
            self.mode = 'follow'
        if state in ('open', 'closed'):
            if self.run is None or self.run[0] != state:
                self.run = (state, now)
            if now - self.run[1] >= self.grip_confirm:
                self.grip = state
        return self.mode


def draw_hand(img, hand, mode):
    """Hand skeleton (green when followed, orange otherwise) and the palm centre."""
    colour = (0, 255, 0) if mode == 'follow' else (0, 165, 255)
    pts = hand.landmarks.astype(int)
    for chain in CHAINS:
        cv2.polylines(img, [pts[chain]], False, colour, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (255, 255, 255), -1)
    c = tuple(hand.center.astype(int))
    cv2.circle(img, c, 7, colour, -1)
    label = f'{hand.state} ({hand.gesture or "-"}, {hand.fingers}/4 fingers)'
    cv2.putText(img, label, (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return img
