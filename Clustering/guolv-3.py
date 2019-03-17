import numpy as np
import csv
import re
import pandas as pd
i1 = 0
i2 = 0
k = 0
A = []
B = []
C = []
D = []
with open('data.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0]=='1' or row[1]=='1':
            A.append(i1)
        i1 = i1 + 1
    A.append(1000000)
with open('data.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0]=='1' or row[1]=='1':
            k = k + 1
        elif row[0] == '0' and row[0] == '0' and k == 0:
            B.append(i2)
        elif row[0]=='0'and row[0]=='0' and (A[k]-A[k-1])>9:
            B.append(i2)
        i2 = i2 + 1

C = np.loadtxt(open("D:\python file\paper\Clustering\TEXT7-3-4.csv","rb"),delimiter=",",skiprows=0)
for j in range(len(B)):
    a = B[j]
    D.append(C[a])


np.savetxt('D:\python file\paper\Clustering\TEXT7-4-2-2.csv',D,delimiter=',')