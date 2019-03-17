import csv
import pandas as pd
import numpy
A = []
B = []
i = 0
j = 0
k = 0
with open('TEXT5-5.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a=float(row[0])
        b=float(row[1])
        k = k + 1
        if a> 2190:
            i = i + 1
            if b == 1:
                j = j + 1

print(i,j,k)
