import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

## 生成示例数据
iris = datasets.load_iris()
X, y = iris.data, iris.target
cv_generator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## OVO
classifier = OneVsOneClassifier(SVC())
results_ovo = cross_validate(classifier, X, y, cv=cv_generator, scoring='accuracy', return_train_score=False)
np.mean(results_ovo['test_score'])

##OVA
classifier = OneVsRestClassifier(SVC())
results_ovr = cross_validate(classifier, X, y, cv=cv_generator, scoring='accuracy', return_train_score=False)
np.mean(results_ovr['test_score'])


#clf2

# from sklearn.datasets import make_blobs
# from sklearn import svm
# import numpy as np
# import torch
# import numpy
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
 
 
# class attDataset(Dataset):
#     def __init__(self,path):
#         super(attDataset,self).__init__()
#         atts=[]
#         with open(path,'r',encoding='utf-8') as f:
#             f = f.readlines()
#             for row in f:
#                 row = row.split()
#                 att = row[:-1]
#                 att = [int(i) for i in att]
#                 label = int(row[-1])
#                 atts.append([att,label])
#             self.atts = atts
        
 
#     def __getitem__(self, index):
#         attribute,label = self.atts[index]
        
#         return attribute,label
#         #return torch.Tensor(attribute),torch.Tensor([label])
 
#     def __len__(self):
#         return len(self.atts)
 
 
# path_train = r'./apascal/attribute_data/attribute_dataset.txt'
# path_test = r'./apascal/attribute_data/attribute_dataset_test.txt'
 
 
 
# train_features = []
# train_label = []
# test_features = []
# test_label = []
 
# train_set =  attDataset(path=path_train)
# test_set  =  attDataset(path=path_test)
 
# for att,label in train_set:
#     train_features.append(att)
#     train_label.append(label)
 
# for att,label in test_set:
#     test_features.append(att)
#     test_label.append(label)
 
# train_features = numpy.array(train_features)
# train_label = numpy.array(train_label)
# test_features = numpy.array(test_features)
# test_label = numpy.array(test_label)
 
 
# clf = svm.SVC(C=5, gamma=0.05,max_iter=200)
# clf.fit(train_features, train_label)
 
 
# #Test on Training data
# train_result = clf.predict(train_features)
# precision = sum(train_result == train_label)/train_label.shape[0]
# print('Training precision: ', precision)
 
 
# #Test on test data
# test_result = clf.predict(test_features)
# precision = sum(test_result == test_label)/test_label.shape[0]
# print('Test precision: ', precision)