import csv
import pandas as pd
import numpy
A = []
B = []
i = 0
with open('TEXT7-4-3.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row in A:
            for j in range(i):
                if A[j] == row:
                    B.append(j)

        else:
            A.append(row)
            B.append(i)
            i = i + 1
l=len(A)
print(i,l)
