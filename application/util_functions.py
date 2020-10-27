import pandas as pd
import numpy as np
from collections import Counter
import csv, pickle, os
from nltk.corpus import stopwords

class_dict = {0:'purpose', 1:'craftsmaship', 2:'aesthetic', 3:"narative", 4:"influence", 5:"none"}

def load_data(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            temp = row
            break
        
        if row[0].lower() != 'labels' or row[1].lower() != 'sentences':            
            print("ERROR: PLZ NAME THE FIRST ROW 'labels' and 'sentences'")
            return
                
        df = pd.read_csv(path)    
        return df


def count_words(features):
    counter = Counter()
    maximum = 0
    
    for sentence in features:
        maximum = max(maximum, len(sentence))
        
        for word in sentence: 
            counter[word] += 1
            
    return maximum, counter


def filter_func(temp):
    
    stop = set(stopwords.words("english"))
    
    temp = temp.lower()
    temp = temp.split()
    temp = [
        element
        for element in temp
        if element not in stop
    ]
    return temp

filter_func = np.vectorize(filter_func, otypes=[list])    


def shuffle(features, labels):
    
    assert labels.shape[0] == features.shape[0]

    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    
    return features[idx], labels[idx]


def onehot_encode_labels(labels):
    index = {
        "purpose"      : [1, 0, 0, 0, 0, 0], 
        "craftsmanship": [0, 1, 0, 0, 0, 0],  
        "aesthetic"    : [0, 0, 1, 0, 0, 0],
        "narative"     : [0, 0, 0, 1, 0, 0],
        "influence"    : [0, 0, 0, 0, 1, 0],
        "none"         : [0, 0, 0, 0, 0, 1]        
    }
    return np.array([
        index[e] 
        for e in labels
    ])

def load_tokenizer():

    with open('application/static/Pickles/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return tokenizer

def save_tokenizer(tokenizer):

    with open('application/static/Pickles/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_classColors():

    with open('application/static/Pickles/class_colors.pickle', 'rb') as handle:
        class_colors = pickle.load(handle)

    return class_colors

def save_classColors(new_purpose, new_craftsmaship, new_aesthetic, new_none):
    class_colors = {'purpose': new_purpose, 'craftsmaship': new_craftsmaship, 'aesthetic': new_aesthetic, 'none':new_none}

    #Overwriting Previous Colors File
    with open('application/static/Pickles/class_colors.pickle', 'wb') as handle:
        pickle.dump(class_colors, handle, protocol=pickle.HIGHEST_PROTOCOL)

def singlefile():
    list = os.listdir("application/static/File_Upload_Folder/")
    #print(list)
    for i in list:
        os.remove("application/static/File_Upload_Folder/"+i)

def decode_onehot_labels(class_arr):
    x = [
        class_dict[class_num] for class_num in class_arr 
    ]

    return x
