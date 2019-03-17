import csv
import re
import pandas as pd
import numpy
A =[]
B =[]
with open('test1.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0] != '0':
            a = row[0]
            A.append(row[0])
        elif row[0] == '0':
            A.append(a)
        if row[1] != '0':
            b = row[1]
            B.append(row[1])
        elif row[1] == '0':
            B.append(b)
numpy.savetxt('test2.csv', A, fmt='%s', delimiter=',')
numpy.savetxt('test3.csv', B, fmt='%s', delimiter=',')


