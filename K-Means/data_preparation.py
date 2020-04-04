import cv2
import glob


class DataPreparation:

    def __init__(self, src, destination, size):
        self.src = src
        self.destination = destination
        self.size = size

    def cut_images(self):
        print("Cutting images into " + str(self.size) +"x"+str(self.size))
        images = glob.glob(self.src)
        img_id = 0
        for image in images:
            img = cv2.imread(image)
            height, width, channels = img.shape
            for h in range(0, height, self.size):
                for w in range(0, width, self.size):
                    crop_img = img[h:h + self.size, w:w + self.size]
                    img_id += 1
                    crop_img_file = str(self.destination) + "/cropped_" + str(img_id) + ".jpg"
                    cv2.imwrite(crop_img_file, crop_img)
        print("--Finished--")