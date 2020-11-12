import os
import numpy as np
import matplotlib.pyplot as plt

train_path = os.path.join('data', "train_images.txt")
train_images = np.loadtxt(train_path, 'uint8')

test_path = os.path.join('data', 'test_images.txt')
test_images = np.loadtxt(test_path, 'uint8')

train_labels_path = os.path.join('data', 'train_labels.txt')
train_labels = np.loadtxt(train_labels_path,'int')

test_labels_path = os.path.join('data', 'test_labels.txt')
test_labels = np.loadtxt(test_labels_path,'int')


from Knn_classifier import Knn_classifier

clasificator = Knn_classifier(train_images,train_labels)

def accuracy(preds):
    return (np.count_nonzero(preds == test_labels)/len(preds))*100

#
# for metrica in ['l1', 'l2']:
#     for k in [1, 3, 5, 7, 9]:
#         pred = clasificator.classify_images(test_images,k,metrica)
#         print('pe metrica = %s si k = %d avem acuratetea %f'%(metrica,k,acuratetea(pred)))

# l2 si k=3 e acuratetea maxima

preds = clasificator.classify_images(test_images,3,'l2')


# confusion matrix

matrix = np.zeros((10,10),dtype=np.int)

for idx,prediction in enumerate(preds):
    matrix[test_labels[idx], prediction] += 1
print("Acuratetea este :", accuracy(preds))
print(np.round((matrix/np.sum(matrix,axis=1))*100))



