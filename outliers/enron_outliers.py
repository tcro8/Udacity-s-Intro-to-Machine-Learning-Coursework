#!/usr/bin/python
import operator
import pickle
import sys
import numpy
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
keys = list(data_dict.keys())
print(keys)
data_dict.pop("TOTAL")
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    print(point)
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
print(numpy.argmax(data))




### your code below



