import csv
import numpy as np
import pandas as pd
import re

i = 0
z = []
A = []
B0 = []
B1 = []
B2 = []
B3 = []
B4 = []
B5 = []
B6 = []
B7 = []
B8 = []
B9 = []
B10 = []
B11 = []
B12 = []
B13 = []
B14 = []
B15 = []
B16 = []
B17 = []
B18 = []
B19 = []
B20 = []
B21 = []
B22 = []
B23 = []
B24 = []
B25 = []
B26 = []
B27 = []
B28 = []
B29 = []
B30 = []
B31 = []
cluster=[]
L = np.loadtxt(open("D:\python file\paper\Clustering\myCentroids.csv","rb"),delimiter=",",skiprows=0)
with open('TEXT7-2-1.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
                a = float(row[4])
                b = float(row[5])
                c = float(row[6])
                d = float(row[7])
                e = float(row[8])
                Lg=[a,b,c,d,e]
                Lg = np.array(Lg)
                L[0]= np.array(L[0])
                M = []
                for j in range(32):
                    dist1= np.sqrt(np.sum(np.square(Lg - L[j])))
                    M.append(dist1)
                k = M.index(min(M))
                if k ==0:
                    B0.append(min(M))
                if k == 1:
                    B1.append(min(M))
                if k ==2:
                    B2.append(min(M))
                if k == 3:
                    B3.append(min(M))
                if k == 4:
                    B4.append(min(M))
                if k == 5:
                    B5.append(min(M))
                if k == 6:
                    B6.append(min(M))
                if k == 7:
                    B7.append(min(M))
                if k == 8:
                    B8.append(min(M))
                if k == 9:
                    B9.append(min(M))
                if k ==10:
                    B10.append(min(M))
                if k == 11:
                    B11.append(min(M))
                if k ==12:
                    B12.append(min(M))
                if k == 13:
                    B13.append(min(M))
                if k == 14:
                    B14.append(min(M))
                if k == 15:
                    B15.append(min(M))
                if k == 16:
                    B16.append(min(M))
                if k == 17:
                    B17.append(min(M))
                if k == 18:
                    B18.append(min(M))
                if k == 19:
                    B19.append(min(M))
                if k ==20:
                    B20.append(min(M))
                if k == 21:
                    B21.append(min(M))
                if k ==22:
                    B22.append(min(M))
                if k == 23:
                    B23.append(min(M))
                if k == 24:
                    B24.append(min(M))
                if k == 25:
                    B25.append(min(M))
                if k == 26:
                    B26.append(min(M))
                if k == 27:
                    B27.append(min(M))
                if k == 28:
                    B28.append(min(M))
                if k == 29:
                    B29.append(min(M))
                if k == 30:
                    B30.append(min(M))
                if k == 31:
                    B31.append(min(M))

A.append(max(B0))
A.append(max(B1))
A.append(max(B2))
A.append(max(B3))
A.append(max(B4))
A.append(max(B5))
A.append(max(B6))
A.append(max(B7))
A.append(max(B8))
A.append(max(B9))
A.append(max(B10))
A.append(max(B11))
A.append(max(B12))
A.append(max(B13))
A.append(max(B14))
A.append(max(B15))
A.append(max(B16))
A.append(max(B17))
A.append(max(B18))
A.append(max(B19))
A.append(max(B20))
A.append(max(B21))
A.append(max(B22))
A.append(max(B23))
A.append(max(B24))
A.append(max(B25))
A.append(max(B26))
A.append(max(B27))
A.append(max(B28))
A.append(max(B29))
A.append(max(B30))
A.append(max(B31))
np.savetxt('PID MAX.csv',A,delimiter=',')

print(A)




