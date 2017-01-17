#!/usr/bin/python
import math
import sys
from time import time
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

from sklearn.metrics import recall_score, precision_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus', 'ratio_to', 'ratio_from', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
### Task 3: Create new feature(s)
def createEmailRatios(data):
    for key in data:
        ## Sent to a POI divided by total sent
        if data[key]['from_this_person_to_poi'] != 'NaN' and data[key]['to_messages'] != 'NaN' and float(data[key]['to_messages']) != 0:
            data[key]['ratio_to'] = float(data[key]['from_this_person_to_poi']) / float(data[key]['to_messages'])
        else:
            data[key]['ratio_to'] = 0

        ## From a POI divided by total recieved
        if data[key]['from_poi_to_this_person'] != 'NaN' and data[key]['from_messages'] != 'NaN' and float(data[key]['from_messages']) != 0:
            data[key]['ratio_from'] = float(data[key]['from_poi_to_this_person']) / float(data[key]['from_messages'])
        else:
            data[key]['ratio_from'] = 0
        
    return data       

    
def createFinancialRatios(data):
    for key in data:
        ## Sent to a POI divided by total sent
        if data[key]['salary'] != 'NaN' and data[key]['bonus'] != 'NaN' and float(data[key]['salary']) != 0:
            data[key]['ratio_bonus_salary'] = float(data[key]['bonus']) / float(data[key]['salary'])
        else:
            data[key]['ratio_bonus_salary'] = 0
    return data
    
data_dict = createEmailRatios(data_dict)
data_dict = createFinancialRatios(data_dict)

### Store to my_dataset for easy export below.
### Scale data for SVM, need to cast all keys as floats to do so


my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Split data into train/test



features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
features_train = preprocessing.scale(features_train)
features_test = preprocessing.scale(features_test)
labels_train
labels_test
    
### Multiple classifiers

print "****SVM*****"
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

parameters = [
  {'C': [1, 10, 100], 'kernel': ['linear']},
  {'C': [1, 100], 'kernel': ['rbf']},
 ]
svr = SVC()
clf_svm = GridSearchCV(svr, parameters)

##clf_svm = SVC(kernel="rbf", C=1000)
t0 = time()
print("Training!")

clf_svm.fit(features_train, labels_train)
print "training time", round(time()-t0, 3), "s"
t1 = time()
pred_svm = clf_svm.predict(features_test)

print "prediction time", round(time()-t1, 3), "s"
print'Accuracy: ', clf_svm.score(features_test, labels_test)
print'Precision: ', precision_score(labels_test, pred_svm)
print'Recall: ', recall_score(labels_test, pred_svm)



print "************"
print ""


print "****NB*****"
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
t0 = time()

clf_nb.fit(features_train, labels_train)
print "training time", round(time()- t0, 3), "s"
t1 = time()
pred_nb = clf_nb.predict(features_test)
print "prediction time", round(time()- t1, 3), "s"
print'Accuracy: ', clf_nb.score(features_test,labels_test)
print'Precision: ', precision_score(labels_test, pred_nb)
print'Recall: ', recall_score(labels_test, pred_nb)
##line break


print "************"
print ""


print "****TREE*****"
from sklearn import tree

clf_tree = tree.DecisionTreeClassifier(criterion='entropy')
t0 = time()
clf_tree.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
acc = clf_tree.score(features_test, labels_test)
pred_tree = clf_tree.predict(features_test)
print'Accuracy: ', acc
print'Precision: ', precision_score(labels_test, pred_tree)
print'Recall: ', recall_score(labels_test, pred_tree)




print "************"
print ""


print "****RANDOMTREE*****"
from sklearn.ensemble import RandomForestClassifier

clf_random_tree = RandomForestClassifier()
t0 = time()
clf_random_tree.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
acc = clf_random_tree.score(features_test, labels_test)
pred_random_tree = clf_random_tree.predict(features_test)
print'Accuracy: ', acc
print'Precision: ', precision_score(labels_test, pred_random_tree)
print'Recall: ', recall_score(labels_test, pred_random_tree)



print "************"
print ""


test_classifier(clf_random_tree, my_dataset, features_list, folds=1000)
test_classifier(clf_tree, my_dataset, features_list, folds=1000)
test_classifier(clf_nb, my_dataset, features_list, folds=1000)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

##dump_classifier_and_data(clf, my_dataset, features_list)