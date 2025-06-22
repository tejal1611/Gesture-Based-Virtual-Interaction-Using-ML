import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
import sys
import os


class AirCanvasApp:
    def __init__(self):
        self.canvas = None
        self.color_palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
        self.current_color = (0, 0, 255)
        self.drawing = False
        self.last_point = None
        self.frame_counter = 0

        # Save button state
        self.save_clicked = False
        self.save_click_time = 0

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

    def draw_palette(self, frame):
        start_x = (frame.shape[1] - (len(self.color_palette) * 50 + 50)) // 2
        for idx, color in enumerate(self.color_palette):
            cv2.rectangle(frame, (start_x + idx * 50, 10), (start_x + 40 + idx * 50, 50), color, -1)
        clear_x_start = start_x + len(self.color_palette) * 50 + 10
        cv2.rectangle(frame, (clear_x_start, 10), (clear_x_start + 40, 50), (50, 50, 50), -1)
        cv2.putText(frame, 'CLR', (clear_x_start + 3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return clear_x_start

    def draw_home_button(self, frame):
        h, w = frame.shape[:2]
        x1, y1 = w - 150, 10  # Top-right corner
        x2, y2 = w - 20, 50
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        cv2.putText(frame, "Home", (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def draw_save_button(self, frame):
        x1, y1 = 20, 10
        x2, y2 = 140, 50
        if self.save_clicked and time.time() - self.save_click_time < 1:
            color = (0, 255, 0)  # Green
        else:
            color = (50, 50, 50)  # Default gray
            self.save_clicked = False
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, "Save", (x1 + 30, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def smooth_line(self, pt1, pt2):
        if pt1 is None or pt2 is None:
            return
        num_points = int(np.linalg.norm(np.array(pt2) - np.array(pt1)) / 2)
        for i in range(1, num_points + 1):
            x = int(pt1[0] + (pt2[0] - pt1[0]) * i / num_points)
            y = int(pt1[1] + (pt2[1] - pt1[1]) * i / num_points)
            cv2.circle(self.canvas, (x, y), 2, self.current_color, -1)

    def is_palm_closed(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        return all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip(finger_tips, finger_pips))

    def handle_gestures(self, hand_landmarks, frame):
        h, w, _ = frame.shape
        index = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        index_x, index_y = int(index.x * w), int(index.y * h)
        thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)

        distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))

        if self.is_palm_closed(hand_landmarks.landmark):
            self.drawing = False
            self.frame_counter = 0
            self.last_point = None
            return

        def is_finger_up(tip_id, pip_id):
            return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

        index_up = is_finger_up(8, 6)
        middle_up = is_finger_up(12, 10)
        ring_up = is_finger_up(16, 14)
        pinky_up = is_finger_up(20, 18)

        if index_up and not (middle_up or ring_up or pinky_up) and distance > 40 / w:
            self.frame_counter += 1
            if self.frame_counter > 3:
                self.drawing = True
                if self.last_point is None:
                    self.last_point = (index_x, index_y)
                else:
                    self.smooth_line(self.last_point, (index_x, index_y))
                    self.last_point = (index_x, index_y)
        else:
            self.drawing = False
            self.frame_counter = 0
            self.last_point = None

        # Color palette selection
        palette_start_x = (frame.shape[1] - (len(self.color_palette) * 50 + 50)) // 2
        clear_start = palette_start_x + len(self.color_palette) * 50 + 10

        if index_y < 60:
            if palette_start_x <= index_x <= palette_start_x + len(self.color_palette) * 50:
                self.current_color = self.color_palette[(index_x - palette_start_x) // 50]
            elif clear_start <= index_x <= clear_start + 40:
                self.canvas[:] = 255

        # Home button logic (top-right corner)
        home_x1, home_y1 = w - 150, 10
        home_x2, home_y2 = w - 20, 50
        if home_x1 <= index_x <= home_x2 and home_y1 <= index_y <= home_y2 and index_up:
            print("Returning to Home...")
            self.launch_home()
            cv2.destroyAllWindows()
            exit()

        # Save button logic (top-left corner)
        save_x1, save_y1 = 20, 10
        save_x2, save_y2 = 140, 50
        if save_x1 <= index_x <= save_x2 and save_y1 <= index_y <= save_y2 and index_up:
            if not self.save_clicked:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"air_canvas_{timestamp}.png"
                cv2.imwrite(filename, self.canvas)
                print(f"Canvas saved as {filename}")
                self.save_clicked = True
                self.save_click_time = time.time()

    def launch_home(self):
        script_path = os.path.join(os.path.dirname(__file__), "menu.py")
        subprocess.Popen([sys.executable, script_path])

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.canvas = np.ones((720, 1280, 3), np.uint8) * 255

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            canvas_copy = self.canvas.copy()
            self.draw_palette(canvas_copy)
            self.draw_home_button(canvas_copy)
            self.draw_save_button(canvas_copy)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(canvas_copy, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.handle_gestures(hand_landmarks, canvas_copy)

            cv2.imshow("Air Canvas", canvas_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.02)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirCanvasApp()
    app.run()
