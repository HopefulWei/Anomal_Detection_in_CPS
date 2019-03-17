import csv
import re
import pandas as pd
i = 0
z = []
with open('test.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        i = i + 1
        c= row[19]
        if 1<i < 219703:
            a = float(row[14])
            if  c =='0':
                z.append(a)
dataframe = pd.DataFrame({'crc rate':z})
dataframe.to_csv("test2.csv",index=False,sep=',')
