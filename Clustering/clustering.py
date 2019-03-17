import csv
import pandas as pd

def clustering(m):

    if 0<=m<=1:
        m=1
    elif 1<=m<2:
        m=2
    elif 2<=m<3:
        m=3
    elif 3 <= m < 4:
        m=4
    elif 4 <= m < 5:
        m=5
    elif 5 <= m < 7:
        m=6
    elif 7 <= m < 9:
        m=7
    elif 9 <= m < 11:
        m=8
    elif 11 <= m < 13:
        m=9
    elif 13 <= m < 15:
        m=10
    elif 15 <= m < 17:
        m=11
    elif 17 <= m < 19:
        m=12
    elif 19 <= m < 21:
        m=13
    elif 21 <= m < 23:
        m=14
    elif 23 <= m < 25:
        m=15
    elif 25 <= m < 29:
        m=16
    elif 29 <= m < 32:
        m=17
    elif 32 <= m < 36:
        m=18
    elif  36<= m <= 40:
        m=19
    elif m == -1:
        m=20
    else:
        m = 21

    return m

def cluster(m):
    if 0<=m<=1:
        m=1
    elif 1<=m<5:
        m=2
    elif 5<=m<9:
        m=3
    elif 9 <= m < 10:
        m=4
    elif 10 <= m < 11:
        m=5
    elif 11 <= m < 12:
        m=6
    elif 12 <= m < 16:
        m=7
    elif 16 <= m < 17:
        m=8
    elif 17 <= m <=20:
        m=9
    elif m == -1:
        m=10
    else:
        m=11

    return m


z = []
y = []
with open('TEXT7-2.csv',"r",encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
            a = row[13]
            b = row[3]
            d = float(a)
            z.append(clustering(d))
            e= float(b)
            y.append(cluster(e))



dataframe = pd.DataFrame({'pressure measurement':z,'setpoint':y})

dataframe.to_csv("test1.csv",index=False,sep=',')






