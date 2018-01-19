import numpy as np
import copy
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression



def rewrite(myfile):
    filename = open(myfile+".revised", "w")
    file = open(myfile, "r")
    count = 0
    for x in file:
        x = x.strip()
        filename.write(x+"\n")        
        count = count + 1
        if count > 5000:
            break
    filename.close()
    file.close()

def getY():
    dataset = np.loadtxt("train", delimiter=",")
    y_train = dataset[:,72].reshape(-1,1)
    return y_train


def read_and_normalize_data():
    rewrite("train_text.vectors")

    dataset = np.loadtxt("train_text.vectors.revised", delimiter=" ")
    x_train = dataset[:,0:4096]
    return x_train

def writeFile(mask):
    filename = open("mask", "w")
    
    for bool in mask:
        filename.write(str(bool))
        filename.write("\n")
    filename.close()


x_train_subarr = read_and_normalize_data()
y_train_subarr = getY()[0:len(x_train_subarr)]

b = SelectKBest(score_func=mutual_info_regression, k=72)
X_new = b.fit_transform(x_train_subarr, y_train_subarr)
mask = b.get_support()
writeFile(mask)
