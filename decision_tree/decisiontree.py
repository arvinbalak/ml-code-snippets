import numpy as np
from anytree import Node, RenderTree
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data = np.genfromtxt('set_a.csv', delimiter=',')

FEATURES = ["S_LENGTH", "S_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"]
FEATURE_INDICES = list(range(4))
CLASSES = ["setosa", "versicolor", "virginica"]
CLASS_INDICES = list(range(3))
CLASS_INDEX = 4

def ID3(features, examples):
    first_class = int(examples[0, CLASS_INDEX])
    if np.all(examples[:,CLASS_INDEX] == first_class):
        return Node(CLASSES[first_class])
    elif len(features) == 0:
        counts = np.bincount((examples[:,CLASS_INDEX]).astype(int))
        return Node(CLASSES[np.argmax(counts)])
    else:
        f, threshold = get_best_feature(features,examples)
        node = Node(FEATURES[f])

        features.remove(f)
        left = ID3(features, examples[np.where(examples[:,f] <= threshold)])
        left.parent = node
        left.edge = "<= {}".format(threshold)

        right = ID3(features, examples[np.where(examples[:,f] > threshold)])
        right.parent = node
        right.edge = "> {}".format(threshold)

        return node

def get_best_feature(features,examples):
    data = np.copy(examples)
    feature_max_info_gain = -np.inf
    max_feature = -1
    max_threshold = 0
    for f in features:
        data_copy = data[data[:,f].argsort()]

        max_info_gain = -np.inf
        max_n = -1
        for n in range(data.shape[0]-1):
            threshold = (data_copy[n,f]+data_copy[n+1,f])/2
            info__gain = calculate_info_gain(f, threshold, data_copy)
            if info__gain > max_info_gain:
                max_info_gain = info__gain
                max_n = n

        if max_info_gain > feature_max_info_gain:
            feature_max_info_gain = max_info_gain
            max_feature = f
            max_threshold = (data_copy[max_n,f]+data_copy[max_n+1,f])/2

    return max_feature, max_threshold

def calculate_info_gain(f, threshold, examples):
    entropy_before = calculate_entropy(examples)

    less_than_examples = examples[examples[:,f] <= threshold]
    greater_than_examples = examples[examples[:, f] > threshold]

    less_than_count = less_than_examples.shape[0]
    greater_than_count = greater_than_examples.shape[0]
    examples_count = examples.shape[0]

    entropy_after = (less_than_count/examples_count * calculate_entropy(less_than_examples)) + \
                    (greater_than_count / examples_count * calculate_entropy(greater_than_examples))

    return (entropy_before - entropy_after)


def calculate_entropy(examples):
    probabilities = []
    sum = examples.shape[0]
    for c in CLASS_INDICES:
        count_c = (examples[:,CLASS_INDEX] == c).sum()
        if sum != 0.0:
            probabilities.append(count_c/sum)

    entropy = 0
    for p in probabilities:
        if p != 0.0:
            entropy -= p * math.log(p, 2)

    return entropy

#tree = ID3(FEATURE_INDICES, data)

#print (RenderTree(tree))

clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(data[:,:4], data[:,4])
tree.export_graphviz(clf, out_file='tree.dot')







