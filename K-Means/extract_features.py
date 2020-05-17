import os
import cv2
import glob
import numpy as np
import statistics
import pandas as pd
import matplotlib.pyplot as plt


class ExtractFeatures():

    def __init__(self, normalize = True):
        self.normalize = normalize

    def image_decomposition(self,image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        '''
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()
        '''

        return np.average(magnitude_spectrum)

    def cornerness(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find Harris corners
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        '''
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # Now draw them
        res = np.hstack((centroids, corners))
        res = np.int0(res)
        img[res[:, 1], res[:, 0]] = [0, 0, 255]
        img[res[:, 3], res[:, 2]] = [0, 255, 0]
        cv2.imshow("s",img)
        '''
        return len(centroids)

    def extract_features(self):
        current = 0
        f = open("dataset.txt", "w")
        main_files = glob.glob("cropped_images/*")
        total = len(main_files)

        header = "filename" + "\t" + "fft" + "\t" + "blobs" + "\t" + "corners" + "\t" + 'b' + "\t" + 'g' + "\t" + 'r' + "\n"
        f.write(header)

        params = cv2.SimpleBlobDetector_Params()

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 1000

        params.filterByCircularity = True
        params.minCircularity = 0.2

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.01

        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        for mainFile in main_files:
            #print("Left:", total - current)
            current += 1
            head, tail = os.path.split(mainFile)
            f.write(tail)
            img = cv2.imread(mainFile)
            avg_magnitude = self.image_decomposition(img)
            f.write("\t" + str(avg_magnitude))

            blobs = self.blobs_detector(img, detector)
            f.write("\t" + str(blobs))

            corners = self.cornerness(img)
            f.write("\t" + str(corners))

            '''
            avg_color_per_row = np.average(img, axis=0)
            avg_color = list(np.average(avg_color_per_row, axis=0))

            for i in range(3):
                f.write("\t" + str(avg_color[i]))
            '''

            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                f.write("\t" + str(hist.max()))

            f.write("\n")
        f.close()

    def normalize_data(self):
        df = pd.read_csv('dataset.txt', sep="\t")
        df.columns = ["filename", "fft", "blobs", "corners", "b", "g", "r"]

        for col in df.columns:
            if col is not "filename":
                df[col] = df[col] / df[col].max()

        df.to_csv("normalized_data", columns=df.columns, index=False)

    def blobs_detector(self,image, detector):

        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect blobs.
        keypoints = detector.detect(im)
        return len(keypoints)

    def blobs_d(self,image, detector):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect blobs.
        keypoints = detector.detect(im)
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(0)

    def extract(self):
        im1 = cv2.imread("cropped_images/cropped_1.jpg") #Laukai
        im2 = cv2.imread("cropped_images/cropped_71.jpg") #Miestas

        self.plot_hist(im1)
        self.plot_hist(im2)


        return
        print("Extracting features...")
        self.extract_features()
        if self.normalize:
            print("Normalizing features...")
            self.normalize_data()
        print("--Finished--")

    def plot_hist(self,im):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([im], [i], None, [256], [0, 256])
            print(str(histr.max()))
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()