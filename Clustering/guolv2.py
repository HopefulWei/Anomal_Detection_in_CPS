import csv
import re
import pandas as pd
import numpy as np
A21=[]
A22=[]
A23=[]
A24=[]
A25=[]

with open('TEXT5-2-2.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c = float(row[5])
        if  c ==0:
            A21.append(row[0])
            A22.append(row[1])
            A23.append(row[2])
            A24.append(row[3])
            A25.append(row[4])



dataframe = pd.DataFrame({'a-21':A21})
dataframe.to_csv("test21.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-22':A22})
dataframe.to_csv("test22.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-23':A23})
dataframe.to_csv("test23.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-24':A24})
dataframe.to_csv("test24.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-25':A25})
dataframe.to_csv("test25.csv",index=False,sep=',')

