import cv2
import numpy as np
from src.config import *
import torch

def display_image(image):
    cv2.imshow("Painting App", 255 - image)
    return cv2.waitKey(10)

def load_model(path):
    if torch.cuda.is_available():
        model = torch.load(path, weights_only=False)
    else:
        model = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    return model

def draw_text(image, text, position, font_scale=1, color=(255, 0, 0), thickness=2):
    """Draw label"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def draw_label_image(background, overlay, position):
    # get dimensions of the overlay
    h, w = overlay.shape[:2]

    # define the region of interest (ROI) on the background image
    x, y = position
    roi = background[y:y + h, x:x + w]

    # directly overlay the image assuming no transparency needed
    overlay_resized = cv2.resize(overlay, (w, h))  # resize overlay to match ROI size
    background[y:y + h, x:x + w] = overlay_resized

    return background

def main():
    # load model
    model = load_model("trained_models/whole_model_quickdraw")

    # set model to evaluation mode
    model.eval()

    # initialize a blank canvas
    image = np.zeros((520, 680, 3), dtype=np.uint8)
    cv2.namedWindow("Painting App")

    # global variables to track drawing status
    global ix, iy, is_drawing
    is_drawing = False

    def paint_draw(event, x, y, flags, param):
        global ix, iy, is_drawing
        # start drawing on left mouse button down
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            ix, iy = x, y
        # draw line while moving the mouse
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing:
                cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
        # stop drawing on left mouse button up
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
            ix = x
            iy = y
        return x, y

    # set callback function for mouse events on the canvas
    cv2.setMouseCallback('Painting App', paint_draw)

    while True:
        # display the inverted canvas
        key = display_image(image)

        # exit loop if 'q' is pressed
        if key == ord("q"):
            break

        # process the drawing when spacebar is pressed
        if key == ord(" "):
            # convert the image to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # find non-zero pixels
            ys, xs = np.nonzero(image_gray)

            if ys.size < 500 or xs.size < 500:
                print("No drawing detected.")
                draw_text(image, "No drawing detected", (10, 50), font_scale=1,
                          color=(255, 255, 35), thickness=2)

                # display the canvas with text
                cv2.imshow("Painting App", 255 - image)
                cv2.waitKey(0)

                # reset the canvas after each prediction
                image = np.zeros((520, 680, 3), dtype=np.uint8)
                continue

            # crop the image to the bounding box of the drawing
            min_y = np.min(ys)
            max_y = np.max(ys)
            min_x = np.min(xs)
            max_x = np.max(xs)

            cropped_image = image_gray[min_y:max_y, min_x: max_x]

            # resize the image to match model input dimensions
            resized_image = cv2.resize(cropped_image, (28, 28))

            # convert image to the format expected by the model
            image_tensor = np.array(resized_image, dtype=np.float32)[None, None, :, :]
            image_tensor = torch.from_numpy(image_tensor)

            # pass the image through the model to get predictions
            logits = model(image_tensor)
            predicted_class = CLASSES[torch.argmax(logits[0])]

            # draw the predicted class on a copy of the canvas
            canvas_with_text = image.copy()
            draw_text(canvas_with_text, "It look like: ", (10, 50), font_scale=1,
                      color=(255, 35, 255), thickness=2)
            draw_text(canvas_with_text, "Press Space to continue", (10, 90), font_scale=1,
                      color=(255, 35, 255), thickness=2)

            # load label image
            label_image = cv2.imread(f"images/{predicted_class}.png")
            print(label_image.shape)
            label_image = cv2.resize(label_image, (50, 50))
            draw_label_image(canvas_with_text, label_image, (205, 15))

            # display the canvas with text
            cv2.imshow("Painting App", 255 - canvas_with_text)
            cv2.waitKey(0)

            # reset the canvas after each prediction
            image = np.zeros((520, 680, 3), dtype=np.uint8)

if __name__ == "__main__":
    main()
