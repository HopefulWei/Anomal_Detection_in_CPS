import csv
import re
import pandas as pd
import numpy
A = []
B = []
C = []

with open('TEXT1.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a = row[0]
        b = row[1]
        c = row[2]
        if a in A:
            d =1
        else:
            A.append(a)
        if b in B:
            d =1
        else:
            B.append(b)
        if c in C:
            d =1
        else:
            C.append(c)

print(A)
print(B)
print(C)

