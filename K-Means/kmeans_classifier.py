import numpy as np
import pandas as pd
import math
import random
import shutil
import os
import time
import sklearn

class KMeansClassifier:
    def __init__(self, clusters_count, dimensions_count, data, columns, epochs):
        self.clusters_count = clusters_count
        self.epochs = epochs
        self.dimensions_count = dimensions_count
        self.data = data
        self.columns = columns
        self.clusters_centroids = []
        self.temp = self.create_empty_clusters_centroids()
        self.clusters_idx = self.create_empty_clusters_idx()
        self.all_atemps = []
        self.generate_clusters()

    '''
    Create empty structure to store selected indexes
    '''
    def create_empty_clusters_idx(self):
        my_list = []
        for i in range(self.clusters_count):
            my_list.append([])
        return my_list

    '''
    Create empty structure to store selected points
    '''
    def create_empty_clusters_centroids(self):
        my_list = []
        for i in range(self.clusters_count):
            my_list.append([])
        return my_list

    '''
    Generate initial centroids
    '''
    def generate_clusters(self):
        self.clusters_centroids = []
        df = pd.DataFrame(self.data, columns=self.columns)
        df = sklearn.utils.shuffle(df)
        for i in range(self.clusters_count):
            self.clusters_centroids.append(self.give_random(df))


    def give_random(self, df):
        generated = False
        num = 0
        while generated is False:
            num = list(df.iloc[random.randint(0, len(df)-1)])
            if len(self.clusters_centroids) == 0:
                return num
            for cluster in self.clusters_centroids:
                if sum(cluster) == sum(num):
                    generated = False
                    break
                else:
                    generated = True
        return num

    '''
    Clusterization
    '''

    def fit(self):
        for epoch in range(self.epochs):
            print("********* Epoch: "+str(epoch)+" **********")
            sec = time.time()
            clusters, idx, variance = self.clusterization(self.create_empty_clusters_idx())
            took_time = time.time() - sec
            print("Time elapsed: " + str(took_time) +" s" )
            print("ETA: ", str(self.epochs*took_time - epoch*took_time) + " s")
            print("Error: ", str(variance))
            print("*****************************")
            print()
            self.all_atemps.append((clusters,idx,variance))
            self.generate_clusters()
        best_epoch = self.select_best()
        self.copy_clusters(best_epoch)
        print("---FINISHED---")
        for i in range(len(best_epoch[1])):
            print("Cluster_"+str(i)+": " + str(len(best_epoch[1][i])) + " items")


    def select_best(self):
        minimum = 999999999
        id = 0
        for idx in range(len(self.all_atemps)):
            if self.all_atemps[idx][2] < minimum:
                minimum = self.all_atemps[idx][2]
                id = idx
        return self.all_atemps[id]


    def clusterization(self, previous_cluster):
        df = pd.DataFrame(self.data, columns=self.columns)
        self.temp = self.create_empty_clusters_centroids()
        self.clusters_idx = self.create_empty_clusters_idx()

        for index, row in df.iterrows():
            self.add_to_centroid(list(row), index)

        if self.clusters_changed(previous_cluster):
            return self.clusterization(self.clusters_idx)
        else:
            return self.temp, self.clusters_idx, self.calculate_variance()


    def clusters_changed(self, previous_cluster):
        changed = False
        for index in range(len(previous_cluster)):
            if sum(previous_cluster[index]) != sum(self.clusters_idx[index]) or len(previous_cluster[index]) != len(self.clusters_idx[index]):
                for i in range(len(self.temp)):
                    cmean = self.calculate_mean_centroids(self.temp[i])
                    self.clusters_centroids[i] = cmean
                changed = True
                break
        return changed

    def calculate_variance(self):
        total = 0
        for idx in range(len(self.clusters_centroids)):
            for point in self.temp[idx]:
                total+=self.calc_euclidean(point,self.clusters_centroids[idx])
        return total





    '''
    Calculate average of each cluster
    '''
    def calculate_mean_centroids(self, row):
        df = pd.DataFrame(row)
        l = list(df.mean(axis=0))
        return l

    '''
    Add data point to the closest centroid
    '''
    def add_to_centroid(self, list1, idx):
        min = 9999999999999
        add_index = 0
        for index in range(len(self.clusters_centroids)):
            distance = self.calc_euclidean(list1, self.clusters_centroids[index])
            if distance < min:
                min = distance
                add_index = index
        self.temp[add_index].append(list1)
        self.clusters_idx[add_index].append(idx)

    '''
    Calculate euclidean distance between two points
    '''
    def calc_euclidean(self, ll1, ll2):
        t = math.sqrt(sum([(a - b) ** 2 for a, b in zip(ll1, ll2)]))
        return t

    '''
    Copy selected clusters to diferrent folders
    '''
    def copy_clusters(self, best_epoch):
        cluster_id = 0
        for cluster in best_epoch[1]:
            os.makedirs("cluster_" + str(cluster_id))
            for idx in cluster:
                shutil.copy("cropped_images/" + str(self.data.iloc[idx]["filename"]), "cluster_" + str(cluster_id))
            cluster_id += 1
