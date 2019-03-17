import csv
import numpy as np
i = 0
j = 0
A = []
B = []
with open("D:\python file\paper\Bloom\setpoint.csv","rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a = float(row[0])
        if 0 < a <  24:
            A.append(0)
        else :
            A.append(1)


with open('data2.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if i > 0:
            b = float(row[0])
            if  A[i - 1] == 1 or b == 1:
                B.append(1)
            elif A[i - 1] == 0 and b == 0:
                B.append(0)
        i = i + 1

np.savetxt("D:\python file\paper\Bloom\Bloom result.csv",B,delimiter=',')
