import cv2
import numpy as np

def get_images(path, classes):
    images = [cv2.imread("{}/{}.png".format(path, item), cv2.IMREAD_UNCHANGED) for item in classes]
    return images

def get_overlay(bg_image, fg_image, sizes=(40, 40)):
    fg_image = cv2.resize(fg_image, sizes)
    fg_mask = fg_image[:, :, 3:]
    fg_image = fg_image[:, :, :3]
    bg_mask = 255 - fg_mask
    bg_image = bg_image/255
    fg_image = fg_image/255
    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)/255
    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)/255
    image = cv2.addWeighted(bg_image*bg_mask, 255, fg_image*fg_mask, 255, 0.).astype(np.uint8)
    return image

if __name__ == '__main__':
    images = get_images("../images", ["bus", "star"])
    print(images[0].shape)
    print(np.max(images[0]))

    cv2.imshow("test image", images[0])
    cv2.waitKey(0)

