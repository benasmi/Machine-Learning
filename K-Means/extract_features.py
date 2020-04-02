import os
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import statistics
import pandas as pd

def image_decomposition(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return np.average(magnitude_spectrum)

'''
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
'''

def extract_features():
    f = open("dataset.txt", "w")
    main_files = glob.glob("cropped_images/*")

    header = "filename" + "\t" + "fft" + "\t" + 'b' + "\t" + 'g' + "\t" + 'r' + "\n"
    f.write(header)

    for mainFile in main_files:
        head, tail = os.path.split(mainFile)
        f.write(tail)
        img = cv2.imread(mainFile)

        avg_magnitude = image_decomposition(img)
        f.write("\t" + str(avg_magnitude))

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            f.write("\t" + str(hist.max()))
        cv2.waitKey(1000)
        f.write("\n")
    f.close()

def normalize_data():
    df = pd.read_csv('dataset.txt', sep="\t")
    df.columns = ["filename", "fft", "b", "g", "r"]
    print(df)
    for col in df.columns:
        if col is not "filename":
            df[col] = df[col] / df[col].max()
    with open('norm_data.txt', 'a') as f:
        f.write(df.to_string(header=True, index=False))
#extract_features()
normalize_data()