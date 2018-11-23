import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pydot

data = np.genfromtxt('set_a.csv', delimiter=',')

FEATURES = ["S_LENGTH", "S_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"]
FEATURE_INDICES = list(range(len(FEATURES)))
CLASSES = ["setosa", "versicolor", "virginica"]
CLASS_INDICES = list(range(len(CLASSES)))
CLASS_INDEX = 4

def ID3(features, examples, used_threshold_n_dict={}):
    first_class = int(examples[0, CLASS_INDEX])
    if np.all(examples[:,CLASS_INDEX] == first_class):
        return Node(CLASSES[first_class], index=first_class)
    elif len(features) == 0:
        counts = np.bincount((examples[:,CLASS_INDEX]).astype(int))
        return Node(CLASSES[np.argmax(counts)], index=np.argmax(counts))
    else:
        f, threshold, n = get_best_feature(features,examples, used_threshold_n_dict)

        if f == -1:
            counts = np.bincount((examples[:, CLASS_INDEX]).astype(int))
            node = Node(CLASSES[np.argmax(counts)], index=np.argmax(counts))
        else:
            used_threshold_n_dict[f] = n
            node = Node(FEATURES[f] + str(threshold))
            node.test_feature = f
            node.threshold = threshold

        if examples[np.where(examples[:,f] < threshold)].shape[0] == 0:
            counts = np.bincount((examples[:, CLASS_INDEX]).astype(int))
            node_child = Node(CLASSES[np.argmax(counts)], index=np.argmax(counts))
            node.parent = node_child
        else:
            left = ID3(features, examples[np.where(examples[:,f] < threshold)], used_threshold_n_dict)
            left.parent = node
            left.edge = "< {}".format(threshold)
            left.test = "lessthan"

            right = ID3(features, examples[np.where(examples[:,f] >= threshold)], used_threshold_n_dict)
            right.parent = node
            right.edge = ">= {}".format(threshold)
            right.test = "greaterthan"

        return node

def get_best_feature(features,examples, used_threshold_n_dict):
    data = np.copy(examples)
    feature_max_info_gain = -np.inf
    max_feature = -1
    max_threshold = 0
    for f in features:
        data_copy = data[data[:,f].argsort()]

        max_info_gain = -np.inf
        max_n = -1
        n_start = 0
        if f in used_threshold_n_dict:
            n_start = used_threshold_n_dict[f]+1
        for n in range(n_start,data.shape[0]-1):
            if data_copy[n,f] != data_copy[n+1,f]:
                threshold = (data_copy[n,f]+data_copy[n+1,f])/2
                info__gain = calculate_info_gain(f, threshold, data_copy)
                if info__gain > max_info_gain:
                    max_info_gain = info__gain
                    max_n = n

        if max_info_gain > feature_max_info_gain:
            feature_max_info_gain = max_info_gain
            max_feature = f
            max_threshold = (data_copy[max_n,f]+data_copy[max_n+1,f])/2

    return max_feature, max_threshold, max_n

def calculate_info_gain(f, threshold, examples):
    entropy_before = calculate_entropy(examples)

    less_than_examples = examples[examples[:,f] < threshold]
    greater_than_examples = examples[examples[:, f] >= threshold]

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

def classify(data, tree):
    result = []
    for i in range(data.shape[0]):
        current_node = tree
        while(not current_node.is_leaf):
            f_to_test = int(current_node.test_feature)
            val = data[i,f_to_test]
            if data[i,f_to_test] < current_node.threshold:
                for child in current_node.children:
                    if child.test == "lessthan":
                        current_node=child
            else:
                for child in current_node.children:
                    if child.test == "greaterthan":
                        current_node=child
        result.append(current_node.index)

    return np.array(result)

# Using hand-made decision tree
handmade_tree = ID3(FEATURE_INDICES, data)
DotExporter(handmade_tree).to_picture("handmade_tree.png")
result = classify(data[:,:CLASS_INDEX],handmade_tree)

success_count = 0
for i in range(data.shape[0]):
    if result[i] == data[i,CLASS_INDEX]:
        success_count+=1
print (success_count/data.shape[0]*100)

# using scikit-learn
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(data[:,:4], data[:,4])
dotfile = StringIO()
tree.export_graphviz(clf, out_file=dotfile)
(graph,)=pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_png("dtree.png")







