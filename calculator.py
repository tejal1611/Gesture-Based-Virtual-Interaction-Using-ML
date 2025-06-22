
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image, ImageTk
import google.generativeai as genai
import tkinter as tk
import subprocess
import sys

# Tkinter GUI Setup
root = tk.Tk()
root.title("Hand Drawing AI Solver")

# Set up Gemini API
genai.configure(api_key="AIzaSyCtjoBxZox1dl1X1Rsm8SSWCh1nAEnp9Cg")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize webcam (high resolution for accurate detection)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(maxHands=1, detectionCon=0.7)

# Canvas and settings
canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
prev_pos = None
output_text = ""

# Constants
BUTTON_X, BUTTON_Y = 650, 20
BUTTON_WIDTH, BUTTON_HEIGHT = 200, 50
CAMERA_FEED_SIZE = (280, 200)
CAMERA_FEED_POS = (980, 20)  # Top-right corner

# Layout
main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
main_pane.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(main_pane, bg="white")
canvas_label = tk.Label(left_frame)
canvas_label.pack(fill=tk.BOTH, expand=True)
main_pane.add(left_frame, minsize=700)

right_frame = tk.Frame(main_pane)
main_pane.add(right_frame, minsize=500)

scrollbar = tk.Scrollbar(right_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

output_textbox = tk.Text(
    right_frame, font=("Arial", 14), wrap=tk.WORD,
    yscrollcommand=scrollbar.set, bg="black", fg="white"
)
output_textbox.pack(fill=tk.BOTH, expand=True)
scrollbar.config(command=output_textbox.yview)

# Function to return to menu
def return_to_menu():
    cap.release()
    cv2.destroyAllWindows()
    subprocess.Popen([sys.executable, "menu.py"])
    root.destroy()

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, drawing_canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(drawing_canvas, current_pos, prev_pos, (255, 0, 255), 10)

    skeleton_overlay = np.ones_like(drawing_canvas) * 255
    skeleton_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    for pair in skeleton_pairs:
        start, end = lmList[pair[0]], lmList[pair[1]]
        cv2.line(skeleton_overlay, (start[0], start[1]), (end[0], end[1]), (0, 0, 255), 2)
    for point in lmList:
        cv2.circle(skeleton_overlay, (point[0], point[1]), 5, (0, 255, 0), cv2.FILLED)

    if fingers == [1, 0, 0, 0, 0]:
        drawing_canvas = np.ones_like(drawing_canvas) * 255

    return current_pos, drawing_canvas, skeleton_overlay

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        canvas = np.uint8(np.clip(canvas, 0, 255))
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""

def update():
    global prev_pos, canvas, output_text

    success, frame = cap.read()
    if not success:
        output_textbox.delete("1.0", tk.END)
        output_textbox.insert(tk.END, "Camera error!")
        return

    frame = cv2.flip(frame, 1)

    display_img = np.copy(canvas)

    # Always show Return button
    cv2.rectangle(display_img, (BUTTON_X, BUTTON_Y),
                  (BUTTON_X + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT),
                  (100, 200, 100), -1)
    cv2.putText(display_img, 'Return to Menu',
                (BUTTON_X + 10, BUTTON_Y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Hand detection
    info = getHandInfo(frame)
    if info:
        fingers, lmList = info
        prev_pos, canvas, skeleton = draw(info, prev_pos, canvas)

        # Overlay skeleton
        display_img = cv2.addWeighted(skeleton, 0.5, display_img, 0.7, 0)

        # Detect finger over button
        index_x, index_y = lmList[8][0:2]
        if (BUTTON_X < index_x < BUTTON_X + BUTTON_WIDTH and
            BUTTON_Y < index_y < BUTTON_Y + BUTTON_HEIGHT):
            return_to_menu()
            return

        # AI query
        response = sendToAI(model, canvas, fingers)
        if response:
            output_text = response

    # Resize camera preview and add to corner
    preview = cv2.resize(frame, CAMERA_FEED_SIZE)
    x, y = CAMERA_FEED_POS
    display_img[y:y + CAMERA_FEED_SIZE[1], x:x + CAMERA_FEED_SIZE[0]] = preview

    # Convert to Tkinter image
    display_img = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=display_img)
    canvas_label.imgtk = imgtk
    canvas_label.configure(image=imgtk)

    # Update text
    output_textbox.delete("1.0", tk.END)
    output_textbox.insert(tk.END, output_text)

    root.after(10, update)

update()
root.mainloop()

