import csv
import re
import pandas as pd
import numpy


i = 0
y = []
c1 = 0
def distance(cluster1,cluster2,element):
    dist1 = numpy.linalg.norm(element - cluster1 )
    dist2 = numpy.linalg.norm(element - cluster2)
    if dist1 > dist2:
        return cluster2
    if dist1 < dist2:
        return cluster1
with open('TEXT7-2.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        i = i + 1
        if 1 <i :
            b = float(row[14])
            clust2 = distance(13231.30530644,17539.95336798, b)
            y.append(clust2)



dataframe = pd.DataFrame({'crc rate':y})
dataframe.to_csv("test3.csv",index=False,sep=',')
