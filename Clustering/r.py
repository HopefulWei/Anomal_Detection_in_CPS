import csv
import re
import pandas as pd
from numpy import *
import numpy
i = 0
y = []
z = []
with open('test.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        i = i + 1
        if row[4] != '?'and row[5] != '?' and row[5] != '?' and row[6] != '?' and row[7] != '?' and row[8] != '?' and i > 1:
            z=[]
            #a = float(row[4])
            #b = float(row[5])
            #c = float(row[6])
            #d = float(row[7])
            a=row[4]
            b=row[5]
            c=row[6]
            d=row[7]
            z.append(a)
            z.append(b)
            z.append(c)
            z.append(d)
            y.append(z)

with open("test4.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(y)
dataframe = pd.DataFrame({'cycle time':v,'deadband':w,'gain':x,'reset rate':y})
dataframe.to_csv("test4.csv",index=False,sep=',',encoding= u'utf-8')

