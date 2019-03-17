import numpy as np
import csv
import re
import pandas as pd
i1 = 0
i2 = 0
i3 = 0
k = 0
A = []
B = []
C = []
D = []
E = []
with open('TEXT7-1-1.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[17] == '1':
            A.append(i1)
        i1 = i1 + 1
    A.append(1000000)
with open('TEXT7-1-1.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[17]=='1':
            k = k + 1
        elif row[17]=='0':
            B.append(i3)
            if A[k] - i3 == 1:
                E.append(1)
            else:
                E.append(0)
        i3 = i3 + 1

C = np.loadtxt(open("D:\python file\paper\Clustering\TEXT7-1-1.csv","rb"),delimiter=",",skiprows=0)
for j in range(len(B)):
    a = B[j]
    D.append(C[a])

dataframe = pd.DataFrame({'segmentation-1':E})
dataframe.to_csv("segmentation-1.csv",index=False,sep=',')

np.savetxt('D:\python file\paper\Clustering\TEXT7-2-1.csv',D,delimiter=',')
