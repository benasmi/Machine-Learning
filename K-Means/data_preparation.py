import cv2
import glob


def cut_images(size, path):
    images = glob.glob(path)
    img_id = 0
    for image in images:
        img = cv2.imread(image)
        height, width, channels = img.shape
        for h in range(0, height, size):
            for w in range(0, width, size):
                crop_img = img[h:h + size, w:w + size]
                img_id += 1
                crop_img_file = "cropped_images/cropped_" + str(img_id) + ".jpg"
                cv2.imwrite(crop_img_file, crop_img)


cut_images(250, "images_to_process/*")
