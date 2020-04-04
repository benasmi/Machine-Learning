from kmeans_classifier import KMeansClassifier
from data_preparation import DataPreparation
from extract_features import ExtractFeatures
import pandas as pd
import statistics as s
import math as m

#Cut images - create folder images_to_process and cropped_images before executing
data_prep = DataPreparation("images_to_process/*","cropped_images/",250)
data_prep.cut_images()

#Extract features
feature_extractor = ExtractFeatures(normalize=True)
feature_extractor.extract()

#Read extracted features
df = pd.read_csv('normalized_data')
usecols = ["fft", "blobs", "corners", "b", "g", "r"]

#Cluster
classifier = KMeansClassifier(clusters_count=2,
                              dimensions_count=5,
                              data=df,
                              columns=usecols,
                              epochs=10)
classifier.fit()




