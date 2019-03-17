import numpy as np
import csv
import re
import pandas as pd
i = 0
A = []
B = []
C = []
with open('TEXT7-1',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0]=='17':
            A.append(i)
        i = i + 1


B = np.loadtxt(open("D:\python file\paper\Clustering\TEXT6-3-4.csv","rb"),delimiter=",",skiprows=0)
for j in range(len(A)):
    a = A[j]
    C.append(B[a])


np.savetxt('D:\python file\paper\Clustering\TEXT6-4-4.csv',C,delimiter=',')













