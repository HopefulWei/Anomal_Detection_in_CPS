import csv
import pandas as pd
import numpy
B=[]
i = 0
with open('LSTM-result-4.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    A=[]
    for row in reader:
        a0=float(row[0])
        a1 = float(row[1])
        a2 = float(row[2])
        a3 = float(row[3])
        a4 = float(row[4])
        A.append(a0)
        A.append(a1)
        A.append(a2)
        A.append(a3)
        if a4 in A:
            B.append(0)
        else:
            B.append(1)
            i = i + 1
print(i)
k1=0
k2=0
k3=0
k4=0
k=0
dataframe = pd.DataFrame({'one hot':B})
dataframe.to_csv("prediction-3.csv",index=False,sep=',')
