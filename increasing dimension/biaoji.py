import csv
import re
import pandas as pd
import numpy
C =[]
F =[]
with open('TEXT.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        A = []
        B = []
        if row[3] != '?':
            for i in range(10):
                j = i + 3
                A.append(row[j])
            A.append('0')
            C.append(A)
        elif row[3] == '?':
            for i in range(10):
                B.append("0")
            B.append('1')
            C.append(B)

        E = []
        D = []
        if row[13] != '?':
            D.append(row[13])
            D.append('0')
            F.append(D)
        elif row[13] == '?':
            E.append('0')
            E.append('1')
            F.append(E)
numpy.savetxt('test3.csv', C, fmt='%s', delimiter=',')
numpy.savetxt('test4.csv',F , fmt='%s', delimiter=',')