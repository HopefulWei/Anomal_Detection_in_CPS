import numpy as np
import csv
import re
import pandas as pd
A =[]
i = 0
with open('data.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a = float(row[0])
        b = float(row[1])
        if a==0 and b ==0:
            A.append(0)
        elif a==1 or b ==1:
            A.append(1)
            i = i + 1
dataframe = pd.DataFrame({'result':A})
dataframe.to_csv("result.csv",index=False,sep=',')
print(i)
