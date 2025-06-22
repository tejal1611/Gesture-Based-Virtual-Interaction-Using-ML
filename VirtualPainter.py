import cv2
import numpy as np
import os
import HandTrackingModule as htm
from tensorflow.keras.models import load_model
import pygame
import time

# Draw rounded rectangles
def draw_rounded_rect(img, top_left, bottom_right, color, radius=10, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    alpha = 0.9
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# Start function
def strt():
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 165, 0)
    PURPLE = (128, 0, 128)
    BACKGROUND = (255, 255, 255)
    BORDER = (0, 255, 0)

    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(3, width)
    cap.set(4, height)
    imgCanvas = np.zeros((height, width, 3), np.uint8)
    drawColor = (0, 0, 255)

    pygame.init()
    FONT = pygame.font.SysFont('freesansbold.ttf', 18)
    DISPLAYSURF = pygame.display.set_mode((width, height), flags=pygame.HIDDEN)
    pygame.display.set_caption("Digit Board")

    number_xcord = []
    number_ycord = []
    label = ""
    PREDICT = "off"
    AlphaMODEL = load_model("bModel.h5")
    NumMODEL = load_model("bestmodel.h5")
    AlphaLABELS = {i: chr(97 + i) for i in range(26)}
    AlphaLABELS[26] = ''
    NumLABELS = {i: str(i) for i in range(10)}
    rect_min_x = rect_max_x = rect_min_y = rect_max_y = 0
    detector = htm.handDetector(detectionCon=0.85)
    xp = yp = 0
    brushThickness = 15
    eraserThickness = 30
    modeValue = "OFF"
    modeColor = RED
    word = ""
    
    home_button_width = 70
    margin_right = 20
    home_x2 = width - margin_right
    home_x1 = home_x2 - home_button_width
    home_y1 = 10
    home_y2 = 50

    palette_start_x = (width - 560) // 2
    color_palette = [
        (palette_start_x + 0, 10, palette_start_x + 70, 50, RED, "Red"),
        (palette_start_x + 80, 10, palette_start_x + 150, 50, YELLOW, "Yellow"),
        (palette_start_x + 160, 10, palette_start_x + 230, 50, GREEN, "Green"),
        (palette_start_x + 240, 10, palette_start_x + 310, 50, BLUE, "BLUE"),
        (palette_start_x + 320, 10, palette_start_x + 390, 50, BLACK, "Eraser"),
        (palette_start_x + 400, 10, palette_start_x + 480, 50, PURPLE, "Save"),
        (palette_start_x + 530, 10, palette_start_x + 600, 50, (255, 0, 255), "Clear"),
        #(palette_start_x + 610, 10, palette_start_x + 680, 50, (0, 128, 255), "Home")
        (home_x1, home_y1, home_x2, home_y2, (0, 128, 255), "Home")

    ]
    
    vertical_x1 = 30
    vertical_x2 = 250
    start_y = 70
    button_height = 40
    spacing = 40

    mode_buttons = [
    (vertical_x1, start_y, vertical_x2, start_y + button_height, GREEN, "Alphabet Mode"),
    (vertical_x1, start_y + (button_height + spacing), vertical_x2, start_y + 2 * button_height + spacing, (139, 0, 0), "Number Mode"),
    (vertical_x1, start_y + 2 * (button_height + spacing), vertical_x2, start_y + 3 * button_height + 2* spacing , RED, "Recognition Off")
    ]

    

    while True:
        SUCCESS, img = cap.read()
        img = cv2.flip(img, 1)
        white_bg = np.ones_like(img) * 255
        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if detector.results.multi_hand_landmarks:
            for handLms in detector.results.multi_hand_landmarks:
                detector.mpDraw.draw_landmarks(white_bg, handLms, detector.mpHands.HAND_CONNECTIONS)
        
        # Replace text mode display with color circle indicator

   # Position and size for the circle
        #circle_center = (700, 70)  # Adjust if needed
        #circle_radius = 15

# Set color based on mode
        if modeValue == 'ALPHABETS':
            modeColor = (0, 255, 0)
            modeText = "Alphabet Mode is ON"  # Green
        elif modeValue == 'NUMBER':
            modeColor = (255, 0, 0)
            modeText = "Number Mode is ON"  # Blue
        else:
            modeColor = (0, 0, 255)  # Red (e.g., no mode)
            modeText = "Recognition is OFF"
        cv2.putText(white_bg, modeText, (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, modeColor, 2)
# Draw the colored circle
        #cv2.circle(white_bg, circle_center, circle_radius, modeColor, -1)

        #Instruction & Mode display
        #cv2.putText(white_bg, f'Mode: {modeValue}', (580, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, modeColor, 2)

        # Draw color palette
        for x1, y1, x2, y2, color, label_text in color_palette:
            draw_rounded_rect(white_bg, (x1, y1), (x2, y2), color, radius=8)
            cv2.putText(white_bg, label_text, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE if color != WHITE else BLACK, 1)

        # Draw Mode Buttons
        for x1, y1, x2, y2, color, label_text in mode_buttons:
            draw_rounded_rect(white_bg, (x1, y1), (x2, y2), color, radius=8)
            cv2.putText(white_bg, label_text, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Hand Gesture Interaction
        if len(lmList) > 0:
            x1, y1 = lmList[8][1:]  # Index finger tip
            fingers = detector.fingersUp()

            # Check for mode button click
            for x1b, y1b, x2b, y2b, color, label_text in mode_buttons:
                if x1b < x1 < x2b and y1b < y1 < y2b:
                    if label_text == "Alphabet Mode" and PREDICT != "alpha":
                        PREDICT = "alpha"
                        modeValue, modeColor = "ALPHABETS", GREEN
                    elif label_text == "Number Mode" and PREDICT != "num":
                        PREDICT = "num"
                        modeValue, modeColor = "NUMBER", YELLOW
                    elif label_text == "Recognition Off" and PREDICT != "off":
                        PREDICT = "off"
                        modeValue, modeColor = "OFF", RED
                        label = ""
                        rect_min_x = rect_max_x = rect_min_y = rect_max_y = 0
                        number_xcord.clear()
                        number_ycord.clear()
                    break  # Optional: Break after match to avoid multiple triggers


            # Check color palette click
            if y1 < 60:
                for x_start, y_start, x_end, y_end, color, label_text in color_palette:
                    if x_start < x1 < x_end and y_start < y1 < y_end:
                        if label_text == "Save":
                            if word.strip() != "":
                                with open("recognized_word.txt", "w") as f:
                                    f.write(word)
                                print("Recognized content saved!")
                            else:
                                print("No content to save.")
                            
                        elif label_text == "Clear":
                            imgCanvas = np.zeros((height, width, 3), np.uint8)
                            word = ""
                            print("Canvas Cleared!")
                        elif label_text == "Home":
                            cap.release()
                            cv2.destroyAllWindows()
                            os.system("python3 menu.py")  # 👈 Launches the menu
                            return
                        else:
                            drawColor = color
                        break

            if fingers[1] and fingers[2] and fingers[3] and fingers[4] and not fingers[0]:
                if len(word) > 0 and word[-1] != " ":
                    word += " "
                    print("Space added!")
                    time.sleep(0.6)

            if fingers[1] and fingers[2]:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)
                if number_xcord and number_ycord and PREDICT != "off":
                    rect_min_x = max(number_xcord[0] - 5, 0)
                    rect_max_x = min(width, number_xcord[-1] + 5)
                    rect_min_y = max(0, number_ycord[0] - 5)
                    rect_max_y = min(number_ycord[-1] + 5, height)
                    number_xcord.clear()
                    number_ycord.clear()

                    img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
                    cv2.rectangle(imgCanvas, (rect_min_x, rect_min_y), (rect_max_x, rect_max_y), BORDER, 3)

                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255

                    if PREDICT == "alpha":
                        label = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(image.reshape(1, 28, 28, 1)))])
                    elif PREDICT == "num":
                        label = str(NumLABELS[np.argmax(NumMODEL.predict(image.reshape(1, 28, 28, 1)))])

                    word += label
                    pygame.draw.rect(DISPLAYSURF, BLACK, (0, 0, width, height))
                    cv2.putText(imgCanvas, label, (rect_min_x, rect_min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

                xp = yp = 0

            elif fingers[1] and not fingers[2]:
                number_xcord.append(x1)
                number_ycord.append(y1)
                cv2.circle(white_bg, (x1, y1 - 15), 15, drawColor, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == BLACK:
                    cv2.line(white_bg, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(white_bg, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    pygame.draw.line(DISPLAYSURF, WHITE, (xp, yp), (x1, y1), brushThickness)

                xp, yp = x1, y1
            else:
                xp = yp = 0

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        white_bg = cv2.bitwise_and(white_bg, imgInv)
        white_bg = cv2.bitwise_or(white_bg, imgCanvas)

        pygame.display.update()
        cv2.imshow("Image", white_bg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry Point
if __name__== "__main__":
    strt()
