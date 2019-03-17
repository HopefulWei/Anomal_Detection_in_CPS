import csv
import re
import pandas as pd
import numpy
i = 0
F=[]
with open('TEXT.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        i = i + 1
        a = row[1]
        if a =='0':
            print(i)
        if a in F:
            a = a
        elif 1:
            F.append(a)
print(F)