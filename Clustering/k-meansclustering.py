import csv
import pandas as pd
import numpy

i = 0
z = []
def distance(cluster1,cluster2,element):
    dist1 = numpy.linalg.norm(element - cluster1 )
    dist2 = numpy.linalg.norm(element - cluster2)
    if dist1 > dist2 and dist2<= 0.0828385*2:
        return 1
    elif dist1 < dist2 and dist1<=0.5610348500000002*2 :
        return 2
    else:
        return 3
with open('TEXT7-2.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a = row[16]
        d = float(a)
        clust1=distance(1.60506515,0.1205615,d)
        z.append(clust1)


dataframe = pd.DataFrame({'time interval':z})
dataframe.to_csv("test3.csv",index=False,sep=',')
