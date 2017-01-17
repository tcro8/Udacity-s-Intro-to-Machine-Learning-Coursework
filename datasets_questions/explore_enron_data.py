#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

size = len(enron_data.keys())
print(size)

features = max(((k, len(v)) for k, v in enron_data.items()), key=lambda x: x[1])

def poi_counter(data):
    num_poi = 0
    for name in data:
        if data[name]["poi"]==1 and data[name]["total_payments"]=="NaN":
            num_poi +=1
        else:
            num_poi = num_poi
    return num_poi
    
for key in enron_data.keys():
    print key
##print(enron_data.keys())

def poi_counter2(data):
    num2_poi = 0
    for name in data:
        if data[name]["total_payments"]=="NaN":
            num2_poi += 1
        else:
            num2_poi = num2_poi
    return num2_poi



print(poi_counter2(enron_data))

  
stock1 = enron_data["SKILLING JEFFREY K"]
print(stock1)
"""
stock3 = enron_data["LAY KENNETH L"]["total_payments"]
print(stock3)

 
stock2 = enron_data["FASTOW ANDREW S"]["total_payments"]
print(stock2)"""
       
print(poi_counter(enron_data))
    


