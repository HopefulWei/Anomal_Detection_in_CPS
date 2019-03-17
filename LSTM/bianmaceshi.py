import csv
import pandas as pd
import numpy
A = []
B = []
i = 0
with open('data1.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a=float(row[0])
        if row[1]=='0':
            A.append(a)
        elif row[1] == '1':
            B.append(a)

l=max(A)
k1=min(B)
k=max(B)
print(l,k,k1)
print(A)
