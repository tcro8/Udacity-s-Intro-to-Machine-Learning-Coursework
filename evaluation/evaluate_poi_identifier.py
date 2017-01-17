#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42 )

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)



for i in range(len(pred)):
    if pred[i] == 1 and labels_test[i] == 1:
        print "Match!"
print(precision_score(labels_test, pred))
print(recall_score(labels_test, pred))
print(pred)
print(len(pred))
print(len(labels_test))
print(clf.score(features_test, labels_test))


### your code goes here 


