


# KD tree

# https://programmer.help/blogs/statistical-learning-method-k-nearest-neighbor-kd-tree-implementation.html
import numpy as np
class Node(object):
    def __init__(self, data, sp = 0 ,left = None, right = None):
        self.data = data # a list
        self.sp = sp # sort by feature sp
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.data  < other.data 

class KDTree(object):
    def __init__(self, data):
        self.dim = data.shape[1] 
        self.root = self.create_tree(data, 0)
        self.nearest_node = None
        self.nearest_dist = np.inf
    
    def create_tree(self, dataset, sp):
        if len(dataset) == 0:
            return None
        dataset_sorted =dataset[np.argsort(dataset[:, sp])]  # sorted by sp dimension on dataset
        mid = dataset.shape[0]//2
        left = self.create_tree(dataset_sorted[:mid], (sp+1)%self.dim)
        right = self.create_tree(dataset_sorted[mid+1:], (sp+1)%self.dim)
        parentNode = Node(dataset_sorted[mid], sp, left, right)
        return parentNode

    def nearest(self, x):
        node = self.root
        self.visit(x, node)
        return self.nearest_node.data, self.nearest_dist
    
    def visit(self, x, node):
        if node != None:
            dis = node.data[node.sp] - x[node.sp]
            self.visit(x, node.left if dis > 0 else node.right)  # if it is None, it would return; then node is Noe' parent
            curr_dis = np.linalg.norm(x-node.data, 2)
            if curr_dis < self.nearest_dist:
                self.nearest_dist = curr_dis
                self.nearest_node = node
            if self.nearest_dist > abs(dis):
                self.visit(x, node.left if dis < 0 else node.right)
        return

# data = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
# kdtree = KDTree(data)  #Create KDTree
# node,dist = kdtree.nearest(np.array([6,5]))
# print(node,dist)

# k_neighbor: find the k_nearest neighbors for input
from math import sqrt

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = np.linalg.norm(np.array(train_row[:2]) - np.array(test_row[:2]))
        distances.append(dist)
    distances = np.argsort(distances)
    neighbors = list()
    for i in range(0,num_neighbors):
        neighbors.append(train[i])
    return neighbors
def get_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key = output_values.count)
    return prediction

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
prediction = get_classification(dataset, dataset[0], 3)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))