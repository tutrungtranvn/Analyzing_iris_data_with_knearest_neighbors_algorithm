#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score , precision_score , roc_auc_score ,roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") 


# In[23]:


iris=pd.read_csv('C:\\Users\\tutru\\Downloads\\iris.csv') # Đọc file iris 


# In[24]:


# Trình chiếu sơ bộ IRIS
iris.head()


# In[25]:


x=iris.iloc[1:,:3] # x chứa các thông tin về chiều dài chiều rộng cánh hoa, đài hoa
y=iris.iloc[1:,4:] # y chứa thông tin của nhãn 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# Lựa chọn số lượng Test Size và Train Size
# Lựa chọn Test Size là 0.2 (30 loài) => Train Size là 0.8 (120 loài)


# In[26]:


print("First five rows")
print(iris.head())
print("*********")
print("columns",iris.columns)
print("*********")
print("shape:",iris.shape)
print("*********")
print("Size:",iris.size)
print("*********")
print("no of samples available for each type")
print(iris['variety'].value_counts())
print("*********")
print(iris.describe())


# In[27]:


print("Training size: %d" %len(y_train))
print("Test size    : %d" %len(y_test))


# In[28]:


cv_scores = [] 
neighbors = list(np.arange(1,50,2)) # tạo một mảng neighbor bất kì để xét
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors = n,algorithm = 'brute') 
    cross_val = cross_val_score(knn,x_train,y_train,cv = 5 , scoring = 'accuracy')
    cv_scores.append(cross_val.mean())    
error = [1-x for x in cv_scores]
optimal_n = neighbors[ error.index(min(error)) ] # Chọn độ k có số loài bị nhiễu thấp nhất
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_n,algorithm = 'brute') # Áp dụng KNeighbors với K tối ưu nhất
knn_optimal.fit(x_train,y_train)
pred = knn_optimal.predict(x_test)
acc = accuracy_score(y_test,pred)*100
print("Độ chính xác tối ưu khi k = {0} là {1}%".format(optimal_n,acc))


# In[29]:


# Phân loại bằng Brute Force
print("Phân loại khi sử dụng Brute Force")
print(classification_report(y_test,pred))


# In[30]:


# Biểu diễn trên ma trận đồ thị
clf = SVC(kernel = 'linear').fit(x_train,y_train)
clf.predict(x_train)
y_pred = clf.predict(x_test)
# Tạo ma trận
cm = confusion_matrix(y_test, y_pred)
# Chuyển đổi sang DataFrame để xét dễ dàng hơn
cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

sns.heatmap(cm_df, annot=True)
acc1 = accuracy_score(y_test,y_pred)*100
plt.title('Độ chính xác:{0}%'.format(acc1))
plt.ylabel('Tên loài hoa thực tế')
plt.xlabel('Tên loài hoa dự đoán')
plt.show()


# In[78]:


def euclidianDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)
def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}
    length = testInstance.shape[1]
    # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        dist = euclidianDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    
    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1)) #by using it we store indices also
    sorted_d1 = sorted(distances.items())
    neighbors = []    
    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])
        counts = {"Setosa":0,"Versicolor":0,"virginica":0}
    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1

    sortedVotes = sorted(counts.items(),key=operator.itemgetter(1), reverse=True)
    print("Xung quanh có : ",sortedVotes)
    return(sortedVotes[0][0], neighbors)


# In[79]:


testSet = [[5.4, 1.6, 3.4, 4.2]]
test = pd.DataFrame(testSet)
result,neigh = knn(iris, test, 20)
print("Hoa thuộc loài :",result)
print("Chỉ số khác biệt: ",neigh)
print("Chỉ số khác biệt ngắn nhất là {0} với chỉ số là {1}".format(result,min(neigh)))






