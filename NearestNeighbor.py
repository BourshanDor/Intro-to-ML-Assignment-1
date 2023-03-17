import math
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np

from sklearn.datasets import fetch_openml
from scipy import stats

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data'] 
labels = mnist['target']

idx = npr.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

train_first_1000 = train[:1000]
train_labels_first_1000 = train_labels[:1000]

def main(): 
        
    #print("the percentage of correct classifications: {}".format(accuracy(test, test_labels, 10, train_first_1000,  train_labels_first_1000 )*100 ))
    #plot_care(test, test_labels, train_first_1000,  train_labels_first_1000)
    q_d(test, test_labels, train, train_labels)

def NK_Alg(train_data, train_labels, k ,image): 
    distance = []
    for i in range(len(train_data)) :
        distance.append(np.linalg.norm(train_data[i] - image))
    distance = np.array(distance)
    arr_correct_position_of_index_k = np.argpartition(distance, k)[:k]
    return stats.mode(train_labels[arr_correct_position_of_index_k])[0][0] 

def accuracy (test_data, test_label, k , train_data, train_labels): 
    accur = []
    i = 0
    for image in test_data : 
        if NK_Alg(train_data, train_labels, k ,image) == test_label[i] : 
            accur.append(1)
        else : 
            accur.append(0) 
        i = i + 1 
    return np.mean(accur)


def plot_care(test_data, test_label, train_data, train_labels): 

    x = [i for i in range(1,101)] 
    y = [accuracy(test_data, test_label, i , train_data, train_labels) for i in x] 
    plt.plot(x, y)
    plt.title("Prediction accuracy as a function of k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()

def q_d(test_data, test_label, train_data, train_labels) : 
    n = [i for i in range(100,5001,100)]
    y = [accuracy(test_data, test_label, 1 , train_data[:N], train_labels[:N]) for N in n] 
    plt.plot(n, y)
    plt.title("Prediction accuracy as a function of n")
    plt.xlabel("n")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    main()