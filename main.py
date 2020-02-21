import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

iris_data = load_iris()
X = iris_data.data
y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
model.fit(X_train,y_train)

def find_rules(tree, features):
    print("find_rules")
    dt = tree.tree_
    def visitor(node, depth):
        indent = ' ' * depth
        if dt.feature[node] != _tree.TREE_UNDEFINED:
            print('{} if <{}> <= {}:'.format(indent, features[node], round(dt.threshold[node], 2)))
            visitor(dt.children_left[node], depth +1)
            print('{}else:'.format(indent))
            visitor(dt.children_right[node],depth +1)
        else:
            print('{}return {}'.format(indent, dt.value[node]))
    visitor(0 ,1)
find_rules(model, iris_data.feature_names)
