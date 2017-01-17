#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    raw_data = zip(predictions,ages,net_worths)
    
       
    cleaned_data = []
    size = len(predictions)

    for p in range(0,size):
        point = (ages[p][0],net_worths[p][0], net_worths[p][0]-predictions[p][0])
        cleaned_data.append(point)
    
    ### your code goes here
    ##for p in cleaned_data:
    ##    print(p)
    ##cleaned_data.sort(key=lambda tup: tup[2])
    def getKey(point):
        return abs(point[2])
    
    cleaned_data = sorted(cleaned_data, key=getKey)
    n = len(cleaned_data)/10
    del cleaned_data[-n:]
    
    print(cleaned_data[:3])
    print(n)
    
        
    
    return cleaned_data

