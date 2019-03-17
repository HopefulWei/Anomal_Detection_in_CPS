import csv
import pandas as pd
import numpy
i1 = 0
i2 = 0
i3 = 0
i4 = 0
with open('prediction-3.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a = float(row[0])
        b = float(row[1])
        if a==b==0:
            i1= i1+ 1
        elif a==b==1:
            i2= i2+ 1
        elif a==0 and b==1:
            i3 = i3 + 1
        elif a==1 and b==0:
            i4 = i4 + 1


print(i1,i2,i3,i4)