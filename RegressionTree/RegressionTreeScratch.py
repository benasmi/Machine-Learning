from __future__ import print_function
import pandas as pd
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import random

def make_sets(df, test_portion):
    train_df = df.sample(frac=test_portion, random_state=random.randint(0,1000))
    test_df = df[~df.index.isin(train_df.index)]

    return train_df, test_df

data = pd.read_csv("train.csv", usecols=["MSZoning", "YearBuilt", "MSSubClass", "LotArea", "LotShape", "Neighborhood", "OverallQual","ExterQual", "TotalBsmtSF", "KitchenQual", "1stFlrSF","GrLivArea","GarageArea","SalePrice"])
data = data.dropna()
header = list(data.columns)

train_data, test_data = make_sets(data,0.8)
print(len(train_data))
print(len(test_data))
print("SDASD")

categorical_cols = ["MSZoning","Neighborhood","ExterQual","KitchenQual", "LotShape"]
numerical_cols = ["YearBuilt","MSSubClass","OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","GarageArea","LotArea"]
split_observation = 10
min_data_in_leaf = 3


def data_analysis_plot(x,y):
    plt.scatter(data[x], data[y], s=1, alpha=1)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def data_analysis_plot_hist(col):
    counts, bins = np.histogram(list(data[col]))
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlabel(col)
    plt.show()

def plot_predictions(err, predictions_points, accuracy):
    actual = []
    predictions = []
    for p in predictions_points:
        actual.append(p[0])
        predictions.append(p[1])
    print(actual)
    print(predictions)
    plt.figure(figsize=(12,5))
    plt.xlabel("Error: " + str(err) + " | " + "Accuracy: " + str(accuracy))
    plt.plot(actual, color="blue", linewidth=0.5, label="Actual price")
    plt.plot(predictions, color='red', linewidth=0.5, label="Predicted price")
    plt.legend(loc="upper left")
    plt.show()


def analyse_all():
    for i in range(len(header)-1):
            try:
                data_analysis_plot(header[i],"SalePrice")
            except:
                print("Ooops")
def uniq_columns(cols):
    return set([col for col in cols])

def analyse_all_plots():
    for i in range(len(numerical_cols)):
            try:
                data_analysis_plot_hist(numerical_cols[i])
            except:
                print("Ooops")

def data_analysis_numerical():
    print("Dataset size %s" % len(data))

    for col in numerical_cols:
        vals = list(data[col])
        print("Col %s has %s values?" % (col,len(uniq_columns(data[col]))))
        print("Min: ", min(vals))
        print("Max:", max(vals))
        print("Median:", statistics.median(vals))
        print("Mean:", statistics.mean(vals))
        print("Std. Dev.:", statistics.stdev(vals))
        print("Q1:", data[col].quantile(0.25))
        print("Q3:", data[col].quantile(0.75))
        print("===============================")

def data_analysis_categorical():
    print("Dataset size %s" % len(data))

    for col in categorical_cols:
        vals = list(data[col])
        print("Col %s has %s values?" % (col,len(uniq_columns(data[col]))))
        mode = statistics.mode(vals)
        print("Mode: ", mode)
        mode_freq = [x for x in vals if x == mode ]
        print("Mode Freq:", len(mode_freq))
        mode = statistics.mode([x for x in vals if x != mode ])
        mode_freq = [x for x in vals if x == mode]
        print("2nd Mode:", mode)
        print("2nd Mode Freq:", len(mode_freq))
        print("===============================")

#data_analysis_categorical()
#data_analysis_numerical()
#analyse_all()
#analyse_all_plots()


def class_counts(data):
    counts = {}
    rows = list(data["SalePrice"])
    for row in rows:
        if row not in counts:
            counts[row] = 0
        counts[row] += 1
    return counts

def residual(outcomes,avg):
    res = 0
    if len(outcomes) > 0:
        for outcome in outcomes:
            res += (outcome-avg)**2
        return res/len(outcomes)
    else:
        print("Nulis")
        return 0

def avg(list):
    if(len(list))>0:
        return sum(list)/len(list)
    else:
        return 0

def is_numeric(column):
    return column in numerical_cols

class Leaf:
    def __init__(self, rows):
        self.predictions = rows
        self.prediction_value = avg(list(rows["SalePrice"]))

class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def __repr__(self):
        condition = "=="
        if self.column in numerical_cols:
            condition = ">="
        return "Is %s %s %s?" % (
            self.column, condition, str(self.value))

    def match(self, example):
        val = example[1][self.column]
        if self.column in numerical_cols:
            try:
              return val >= self.value
            except:
              print(self.column, self.value)
        else:
            try:
                return val == self.value
            except:
                print(self.column, self.value)

def splitData(data,question):
    if is_numeric(question.column):
            if all(p >= question.value for p in list(data[question.column])) or all(p <= question.value for p in list(data[question.column])):
                left = data
                right = pd.DataFrame(columns=header)
            else:
                left, right = [x for _, x in data.groupby(data[question.column] >= question.value)]
    else:
            if all(p == question.value for p in list(data[question.column])):
                left = data
                right = pd.DataFrame(columns=header)
            else:
                left, right = [x for _, x in data.groupby(data[question.column] == question.value)]
    return left, right

def satisfies_split_criteria(data, l, r):
    return (len(data) != len(l) and len(data) != len(r)) and (len(l) >= min_data_in_leaf and len(r) >=min_data_in_leaf)


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(data):
    can_split = True
    question = None
    totalResidual = 0
    currentResidual = math.inf

    if (data.shape[0] <= split_observation):
        can_split = False
    else:
        for x in categorical_cols:
            for y in uniq_columns(data[x]):
                l, r = splitData(data, Question(x,y))
                if satisfies_split_criteria(data,l,r):
                    leftAvg = avg([x/100000 for x in list(l["SalePrice"])])
                    rightAvg = avg([x/100000 for x in list(r["SalePrice"])])
                    totalResidual = residual([x/100000 for x in list(l["SalePrice"])], leftAvg) + residual([x/100000 for x in list(r["SalePrice"])],
                                                                                                    rightAvg)
                    if totalResidual < currentResidual:
                        currentResidual = totalResidual
                        question = Question(x, y)

        for x in numerical_cols:
            values = list(data[x])
            split_point = avg(values)
            l, r = splitData(data, Question(x, split_point))
            if satisfies_split_criteria(data, l, r):
                leftAvg = avg([x / 100000 for x in list(l["SalePrice"])])
                rightAvg = avg([x / 100000 for x in list(r["SalePrice"])])
                totalResidual = residual([x / 100000 for x in list(l["SalePrice"])], leftAvg) + residual(
                        [x / 100000 for x in list(r["SalePrice"])],
                        rightAvg)
                if totalResidual < currentResidual:
                    currentResidual = totalResidual
                    question = Question(x, split_point)

        '''
        for x in numerical_cols:
                values = list(data[x])
                values.sort()
                for i in range(len(values)-1):
                    split_point = avg(list([values[i], values[i+1]]))
                    l, r = splitData(data,Question(x, split_point))
                    if satisfies_split_criteria(data,l,r):
                        leftAvg = avg([x / 100000 for x in list(l["SalePrice"])])
                        rightAvg = avg([x / 100000 for x in list(r["SalePrice"])])
                        totalResidual = residual([x / 100000 for x in list(l["SalePrice"])], leftAvg) + residual(
                            [x / 100000 for x in list(r["SalePrice"])],
                            rightAvg)
                        if totalResidual < currentResidual:
                            currentResidual = totalResidual
                            question = Question(x,split_point)
        '''
        if question == None:
            print("What")
    return can_split, question


def build_tree(rows):

    canSplit, question = find_best_split(rows)

    if not canSplit:
        print("Leaf")
        return Leaf(rows)

    print(question)
    left, right = splitData(rows, question)
    print("Q:",(question,len(right),len(left)))
    true_branch = build_tree(left)
    false_branch = build_tree(right)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        print("Exact value: " + str(node.prediction_value))
        return

    print (spacing + str(node.question))

    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def tree_error(validation_data, tree):
    err = 0
    percentage = 0
    prediction_points = []
    for row in validation_data.iterrows():
        actual = row[1]["SalePrice"]
        predicted = classify(row, tree)[1]
        err = err + (actual - predicted)**2
        prediction_points.append((actual,predicted))
        percentage = percentage + math.fabs((actual - predicted)/actual * 100)
    return math.sqrt(err/len(validation_data)), prediction_points, (100-percentage/len(validation_data))


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions, node.prediction_value

    if node.question.match(row):
        return classify(row, node.false_branch)
    else:
        return classify(row, node.true_branch)



def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



my_tree = build_tree(train_data)
print_tree(my_tree)

loss, predictions, accuracy = tree_error(test_data,my_tree)
plot_predictions(loss,predictions, accuracy)



for row in test_data.iterrows():
        actual = row[1]["SalePrice"]
        print ("Actual: %s. Predicted: %s" %
               (actual, classify(row, my_tree)[1]))












