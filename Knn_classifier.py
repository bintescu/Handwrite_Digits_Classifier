import numpy as np

def euclidian_distance(img1, img2):
    return np.sqrt(np.sum((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def manhattan_distance(img1, img2):
    return np.sum(np.abs(img1.astype(np.float64) - img2.astype(np.float64)))


class Knn_classifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels


    def first_K_neighbors(self, test_image, num_neighbors = 3, metric = 'l2'):
        all_distances = []
        if metric == 'l1':
            for image in self.train_images:
                all_distances.append(manhattan_distance(test_image,image))
        elif metric == 'l2':
            for image in self.train_images:
                all_distances.append(euclidian_distance(test_image,image))
        else:
            print("Valoare introdusa pentru metrica este gresita !")
        #indexes = np.argpartition(np.asarray(all_distances),num_neighbors) #extrage indexul primelor cele mai mici num_neighbors
        indexes = np.argsort(all_distances)[:num_neighbors]
        return indexes

    def classify_image(self,test_image,num_neighbors = 3, metric = 'l2'):
        all_neighbors_indexes = self.first_K_neighbors(test_image,num_neighbors,metric)
        all_neighbors = self.train_labels[all_neighbors_indexes]
        neighbors = np.bincount(all_neighbors)
        while( np.sum((neighbors == np.max(neighbors))) > 1):
            num_neighbors += 1
            all_neighbors_indexes = self.first_K_neighbors(test_image,num_neighbors , metric)
            all_neighbors = self.train_labels[all_neighbors_indexes]
            neighbors = np.bincount(all_neighbors)
        return neighbors.argmax()

    def classify_images(self, imagini_test, k=3, metric='l2'):
        '''Returneaza toate predictiile pt toate imaginile de test.
        '''
        preds = np.zeros(imagini_test.shape[0], dtype=np.int)
        for idx,img in enumerate(imagini_test):
            preds[idx] = self.classify_image(img, num_neighbors=k, metric=metric)
        return preds