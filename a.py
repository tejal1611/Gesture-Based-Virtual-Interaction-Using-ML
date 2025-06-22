import cv2
import mediapipe as mp
import numpy as np
import subprocess
import sys
import os

# Tool scripts
MODULES = {
    "Air Canvas": "canvas.py",
    "Virtual Calculator": "calculator.py",
    "Document Interaction": "document.py",
    "Hand-to-Text": "VirtualPainter.py"
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Finger detection
def fingers_up(lm_list):
    tips = [8, 12, 16, 20]
    up = []
    up.append(1 if lm_list[tips[0]][1] < lm_list[tips[0] - 2][1] else 0)
    for id in range(1, 4):
        up.append(1 if lm_list[tips[id]][2] < lm_list[tips[id] - 2][2] else 0)
    return up

# Launch module
def run_module(script):
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), script)
    subprocess.Popen([python_exe, script_path])

# Button class
class Button:
    def __init__(self, pos, text, size=(300, 60), color=(220, 220, 220), active_color=(100, 255, 100)):
        self.pos = pos
        self.text = text
        self.size = size
        self.color = color
        self.active_color = active_color

    def draw(self, img, hover=False):
        col = self.active_color if hover else self.color
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, (x, y), (x + w, y + h), col, cv2.FILLED)
        cv2.putText(img, self.text, (x + 10, y + h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def is_hover(self, pt):
        x, y = self.pos
        w, h = self.size
        return x < pt[0] < x + w and y < pt[1] < y + h

# Prepare buttons
buttons = [Button((50, 100 + i * 80), name) for i, name in enumerate(MODULES)]
exit_button = Button((600, 38), "X", size=(50, 50), color=(100, 100, 100), active_color=(0, 0, 255))

cap = cv2.VideoCapture(0)
selected_tool = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    img = 255 * np.ones_like(frame)

    # Process hand
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    index_tip = None
    one_finger = False

    if res.multi_hand_landmarks and selected_tool is None:
        lm = res.multi_hand_landmarks[0]
        pts = [[id, int(lm.landmark[id].x * w), int(lm.landmark[id].y * h)] for id in range(21)]
        
        # Finger gesture: only index finger up
        if fingers_up(pts) == [1, 0, 0, 0]:
            one_finger = True
            index_tip = (pts[8][1], pts[8][2])  # Index fingertip coordinates

        mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec((0, 0, 255), 2, 4),
                               mp_draw.DrawingSpec((0, 255, 0), 2, 2))

    # Draw fingertip for debugging
    if index_tip:
        cv2.circle(img, index_tip, 10, (255, 0, 0), -1)

    # Draw all tool buttons
    for btn in buttons:
        hover = one_finger and index_tip and btn.is_hover(index_tip)
        btn.draw(img, hover)
        if hover and selected_tool is None:
            print(f"[INFO] Launching: {btn.text}")
            selected_tool = btn.text
            cap.release()
            cv2.destroyAllWindows()
            run_module(MODULES[btn.text])
            os._exit(0)  # Hard exit

    # Draw and check exit button
    hover_exit = one_finger and index_tip and exit_button.is_hover(index_tip)
    exit_button.draw(img, hover_exit)

    if hover_exit:
        print(f"[INFO] Exiting - Hover detected at {index_tip}")
        cap.release()
        cv2.destroyAllWindows()
        os._exit(0)  # Hard exit

    # Message
    if selected_tool:
        cv2.putText(img, f"Opening {selected_tool}...", (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)

    cv2.imshow("Gesture Menu", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

