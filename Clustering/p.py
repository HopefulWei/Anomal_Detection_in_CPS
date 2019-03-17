import csv
import re
import pandas as pd
i = 0
y = []
b1 = 0
with open('time.txt',"r") as txtfile:
    for row in txtfile:
            b = float(row)
            d = b - b1
            y.append(d)
            b1 = b
dataframe = pd.DataFrame({'time interval':y})
dataframe.to_csv("test5.csv",index=False,sep=',')