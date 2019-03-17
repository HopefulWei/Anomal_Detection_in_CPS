import pandas as pd
import numpy
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(train)
array = enc.transform(train).toarray()

#对分类类型的数据进行ONEHOT编码
train = pd.read_csv('D:\python file\paper\increasing dimension\label-3.csv',delimiter=',')
pd.np.isnan(train).any()
train.dropna(inplace=True)

A,B,C,D=[],[],[],[]
for i in range(117735):
    a=array[i]
    A.append(a)
for i in range(2000):
    c = array[i]
    C.append(c)
for i in range(117735,len(array)):
    b=array[i]
    B.append(b)
for i in range(117735,117835):
    d=array[i]
    D.append(d)



numpy.savetxt('train3-1.csv',A, delimiter = ',')
numpy.savetxt('train3-2.csv',C, delimiter = ',')
numpy.savetxt('test3-1.csv',B, delimiter = ',')
numpy.savetxt('test3-2.csv',D, delimiter = ',')
