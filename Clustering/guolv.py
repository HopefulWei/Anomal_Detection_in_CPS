import csv
import re
import pandas as pd
import numpy as np
A0=[]
A1=[]
A2=[]
A3=[]
A4=[]
A5=[]
A6=[]
A7=[]
A8=[]
A9=[]
A10=[]
A11=[]
A12=[]
A13=[]
A14=[]
A15=[]
A16=[]
A17=[]
A18=[]
A19=[]
A20=[]
A21=[]
A22=[]
i = 0

with open('TEXT6-1-1.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        i = i + 1
        c = float(row[17])
        if  c ==0 :
            A0.append(row[0])
            A1.append(row[1])
            A2.append(row[2])
            A3.append(row[3])
            A4.append(row[4])
            A5.append(row[5])
            A6.append(row[6])
            A7.append(row[7])
            A8.append(row[8])
            A9.append(row[9])
            A10.append(row[10])
            A11.append(row[11])
            A12.append(row[12])
            A13.append(row[13])
            A14.append(row[14])
            A15.append(row[15])
            A16.append(row[16])
            A17.append(row[17])
            A18.append(row[18])
            A19.append(row[19])


dataframe = pd.DataFrame({'a-0':A0})
dataframe.to_csv("test0.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-1':A1})
dataframe.to_csv("test1.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-2':A2})
dataframe.to_csv("test2.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-3':A3})
dataframe.to_csv("test3.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-4':A4})
dataframe.to_csv("test4.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-5':A5})
dataframe.to_csv("test5.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-6':A6})
dataframe.to_csv("test6.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-7':A7})
dataframe.to_csv("test7.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-8':A8})
dataframe.to_csv("test8.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-9':A9})
dataframe.to_csv("test9.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-10':A10})
dataframe.to_csv("test10.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-11':A11})
dataframe.to_csv("test11.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-12':A12})
dataframe.to_csv("test12.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-13':A13})
dataframe.to_csv("test13.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-14':A14})
dataframe.to_csv("test14.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-15':A15})
dataframe.to_csv("test15.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-16':A16})
dataframe.to_csv("test16.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-17':A17})
dataframe.to_csv("test17.csv",index=False,sep=',')


dataframe = pd.DataFrame({'a-18':A18})
dataframe.to_csv("test18.csv",index=False,sep=',')

dataframe = pd.DataFrame({'a-19':A19})
dataframe.to_csv("test19.csv",index=False,sep=',')
















