import csv
import re
import pandas as pd
import numpy

i = 0
z = []
A = []
B = []
C = []
D = []
E = []
cluster=[]
with open('TEXT7-2.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
                a = float(row[4])
                b = float(row[5])
                c = float(row[6])
                d = float(row[7])
                e = float(row[8])
                Lg=[a,b,c,d,e]
                L=[[ -1,-1,-1,-1,-1]
 [119, 0.29574049, 0.46887117, 0.81615951, 0]
 [114, 0.28954504, 0.45679245, 1.26639869, 0]
 [112, 0.29201158, 0.65913449, 1.256498, 0]
 [116, 0.27432796, 0.62630824, 1.27143369, 0]
 [117, 0.32403344, 0.47673356, 0.78426979, 0]
 [110, 0.29638208, 0.67117257, 0.74442478, 0]
 [113, 0.31213992, 0.34804622, 1.26279412, 0]
 [111, 0.27968098, 0.67230366, 1.26734729, 0]
 [118, 0.3144324, 0.34816415, 1.2987041, 0]
 [115, 0.30280351, 0.66920664, 1.27254613, 0]
 [110, 0.27417488, 0.6609688, 1.21114943, 0]
 [111, 0.30371955, 0.32464223, 0.79657941, 0]
 [113, 0.3118166, 0.54807157, 0.8433996, 0]
 [119, 0.30029037, 0.38111111, 1.26145185, 0]
 [117, 0.30462118, 0.45470823, 1.2526299, 0]
 [114, 0.3188267, 0.42395928, 0.81671946, 0]
 [115, 0.3023792, 0.48436893, 0.78800277, 0]
 [118, 0.28802522, 0.31136499, 0.82083086, 0]
 [112, 0.30288513, 0.49669276, 0.76996086, 0]
 [116, 0.30552827, 0.52340256, 0.80445687, 0]
 [113, 0.31043004, 0.62148387, 1.27022222, 0]
 [111, 0.29755741, 0.35025367, 1.23738318, 0]
 [119, 0.29270156, 0.66951143, 1.27502079, 0]
 [110, 0.27842221, 0.38210475, 1.2714064, 0]
 [110, 0.29105274, 0.34987226, 0.79284672, 0]
 [112, 0.28121479, 0.3888746, 1.27882637, 0]
 [115, 0.29457124, 0.39329041, 1.2497711, 0]
 [118, 0.30428708, 0.62953846, 1.25801538, 0]
 [111, 0.29926777, 0.62055838, 0.73218274, 0]
 [118, 0.30228047, 0.64889182, 0.77699208, 0]
 [116, 0.29009838, 0.28425926, 1.28724537, 0]]




                Lg = numpy.array(Lg)
                L[0]= numpy.array(L[0])
                M = []
                for j in range(32):
                    dist1= numpy.sqrt(numpy.sum(numpy.square(Lg - L[j])))
                    M.append(dist1)
                k = M.index(min(M))

                cluster1 = L[k][0]
                cluster2 = L[k][1]
                cluster3 = L[k][2]
                cluster4 = L[k][3]
                cluster5 = L[k][4]
                A.append(cluster1)
                B.append(cluster2)
                C.append(cluster3)
                D.append(cluster4)
                E.append(cluster5)



dataframe = pd.DataFrame({'PID Clustering1':A,'PID Clustering2':B,'PID Clustering3':C,'PID Clustering4':D,'PID Clustering5':E})
dataframe.to_csv("test4.csv",index=False,sep=',')
