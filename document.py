import cv2
import numpy as np
import mediapipe as mp
from tkinter import filedialog, Tk
import fitz  # PyMuPDF
import time
import tempfile
import os
import shutil
import subprocess
import sys

class AirCanvasPDFViewer:
    def __init__(self):
        self.width, self.height = 1280, 720
        self.button_color = (200, 200, 200)
        self.text_color = (0, 0, 0)
        self.clear_button_color = (255, 100, 100)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

        self.pdf_uploaded = False
        self.current_page = 0
        self.pdf_document = None
        self.canvas_pages = {}
        self.default_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.prev_x, self.prev_y = None, None
        self.prev_palm_y = None
        self.scroll_cooldown = time.time()
        self.scroll_delay = 1.0
	
    def upload_pdf(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        return file_path if file_path else None

    def load_pdf(self, file_path):
        self.pdf_document = fitz.open(file_path)
        self.pdf_uploaded = True
        self.current_page = 0
        self.canvas_pages = {}

    def render_pdf_page(self):
        if self.pdf_document:
            page = self.pdf_document.load_page(self.current_page)
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB if pix.n == 4 else cv2.COLOR_GRAY2RGB if pix.n == 1 else cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (self.width, self.height))
            return img
        return np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

    def draw_buttons(self, frame):
        # Upload PDF
        cv2.rectangle(frame, (200, 20), (400, 70), self.button_color, -1)
        cv2.putText(frame, 'Upload PDF', (220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)

        # Clear
        cv2.rectangle(frame, (self.width // 2 - 100, 20), (self.width // 2 + 100, 70), self.clear_button_color, -1)
        cv2.putText(frame, 'Clear', (self.width // 2 - 40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)

        # Save
        cv2.rectangle(frame, (self.width - 400, 20), (self.width - 200, 70), self.button_color, -1)
        cv2.putText(frame, 'Save', (self.width - 340, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2)

        # Return to Menu
        cv2.rectangle(frame, (20, self.height - 70), (220, self.height - 20), (100, 200, 100), -1)
        cv2.putText(frame, 'Home', (30, self.height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def scroll_pdf(self, palm_y, palm_closed):
        if self.pdf_document and time.time() - self.scroll_cooldown > self.scroll_delay:
            if self.prev_palm_y is not None and palm_closed:
                movement = palm_y - self.prev_palm_y
                if movement > 50 and self.current_page < len(self.pdf_document) - 1:
                    self.current_page += 1
                    self.scroll_cooldown = time.time()
                elif movement < -50 and self.current_page > 0:
                    self.current_page -= 1
                    self.scroll_cooldown = time.time()
            self.prev_palm_y = palm_y

    def is_palm_closed(self, hand_landmarks):
        return sum(hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 2].y for i in [8, 12, 16, 20]) >= 3

    def save_pdf_with_highlights(self):
        if self.pdf_document:
            try:
                save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
                if save_path:
                    new_pdf = fitz.open()
                    temp_dir = tempfile.mkdtemp()

                    for i in range(len(self.pdf_document)):
                        page = self.pdf_document.load_page(i)
                        pix = page.get_pixmap()
                        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB if pix.n == 4 else cv2.COLOR_GRAY2RGB if pix.n == 1 else cv2.COLOR_RGB2BGR)
                        img = cv2.resize(img, (self.width, self.height))

                        if i in self.canvas_pages:
                            highlight = self.canvas_pages[i]
                            img = cv2.addWeighted(img, 1, highlight, 1, 0)

                        temp_img_path = os.path.join(temp_dir, f"temp_page_{i}.png")
                        cv2.imwrite(temp_img_path, img)

                        rect = fitz.Rect(0, 0, self.width, self.height)
                        new_page = new_pdf.new_page(width=self.width, height=self.height)
                        new_page.insert_image(rect, filename=temp_img_path)

                    new_pdf.save(save_path)
                    new_pdf.close()
                    shutil.rmtree(temp_dir)
                    print(f"PDF saved with highlights: {save_path}")
            except Exception as e:
                print(f"Error saving PDF: {e}")

    def clear_highlights(self):
        if self.pdf_uploaded:
            if self.current_page in self.canvas_pages:
                self.canvas_pages[self.current_page] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            self.default_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def fingers_up(self, hand_landmarks):
        fingers = [1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0]
        for i in [8, 12, 16, 20]:
            fingers.append(1 if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y else 0)
        return fingers

    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)

            final_frame = self.render_pdf_page() if self.pdf_uploaded else frame.copy()
            self.draw_buttons(final_frame)

            if self.pdf_uploaded and self.current_page not in self.canvas_pages:
                self.canvas_pages[self.current_page] = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            canvas = self.canvas_pages[self.current_page] if self.pdf_uploaded else self.default_canvas

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(final_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    index_tip = hand_landmarks.landmark[8]
                    palm_y = int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * self.height)
                    index_x, index_y = int(index_tip.x * self.width), int(index_tip.y * self.height)

                    fingers = self.fingers_up(hand_landmarks)

                    if fingers[1] == 1 and all(f == 0 for f in fingers[2:]):
                        if self.prev_x is not None and self.prev_y is not None:
                            overlay = canvas.copy()
                            cv2.line(overlay, (self.prev_x, self.prev_y), (index_x, index_y), (0, 0, 255), 25)
                            canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)
                        self.prev_x, self.prev_y = index_x, index_y
                    else:
                        self.prev_x, self.prev_y = None, None

                    if self.pdf_uploaded:
                        self.canvas_pages[self.current_page] = canvas
                    else:
                        self.default_canvas = canvas

                    # Upload PDF
                    if 200 < index_x < 400 and 20 < index_y < 70:
                        file_path = self.upload_pdf()
                        if file_path:
                            self.load_pdf(file_path)

                    # Save PDF
                    if self.pdf_uploaded and self.width - 400 < index_x < self.width - 200 and 20 < index_y < 70:
                        self.save_pdf_with_highlights()

                    # Clear Highlights
                    if self.width // 2 - 100 < index_x < self.width // 2 + 100 and 20 < index_y < 70:
                        self.clear_highlights()

                    # Scroll PDF
                    if self.pdf_uploaded and self.is_palm_closed(hand_landmarks):
                        self.scroll_pdf(palm_y, self.is_palm_closed(hand_landmarks))

                    # Return to Menu
                    if 20 < index_x < 220 and self.height - 70 < index_y < self.height - 20:
                        cap.release()
                        cv2.destroyAllWindows()
                        subprocess.Popen([sys.executable, "menu.py"])
                        return

            combined = cv2.addWeighted(final_frame, 1, canvas, 1, 0)
            cv2.imshow('Air Canvas PDF Viewer with Highlight', combined)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('c'):
                self.clear_highlights()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    viewer = AirCanvasPDFViewer()
    viewer.run()

