import csv
import pandas as pd
import numpy

i = 0
A = []
B = []
def distance(cluster1,cluster2,element):
    dist1 = numpy.linalg.norm(element - cluster1 )
    dist2 = numpy.linalg.norm(element - cluster2)
    if dist1 > dist2:
        B.append(dist2)
    if dist1 < dist2:
        A.append(dist1)
with open('TEXT7-2-1.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a = row[16]
        d = float(a)
        clust1=distance(1.60506515,0.1205615,d)
print(max(A))
print(max(B))

