import csv
import pandas as pd
import numpy  as np

i = 0
z = []
A = []
L = np.loadtxt(open("D:\python file\paper\Clustering\myCentroids.csv","rb"),delimiter=",",skiprows=0)
Y = np.loadtxt(open("D:\python file\paper\Clustering\PID MAX.csv","rb"),delimiter=",",skiprows=0)
cluster=[]
with open('D:\python file\paper\Clustering\TEXT7-2.csv',"r", encoding='UTF-8')as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
                a = float(row[4])
                b = float(row[5])
                c = float(row[6])
                d = float(row[7])
                e = float(row[8])
                Lg=[a,b,c,d,e]

                Lg = np.array(Lg)
                L[0]= np.array(L[0])
                M = []
                for j in range(32):
                    dist1= np.sqrt(np.sum(np.square(Lg - L[j])))
                    M.append(dist1)
                k = M.index(min(M))
                if k == 0 and min(M)==Y[0]:
                    A.append(0)
                elif k == 1 and min(M)<=Y[1]*1.2:
                    A.append(1)
                elif k == 2 and min(M)<=Y[2]*1.2:
                    A.append(2)
                elif k == 3 and min(M)<=Y[3]*1.2 :
                    A.append(3)
                elif k == 4 and min(M)<=Y[4]*1.2:
                    A.append(4)
                elif k == 5 and min(M)<=Y[5]*1.2:
                    A.append(5)
                elif k == 6 and min(M)<=Y[6] *1.2:
                    A.append(6)
                elif k == 7 and min(M)<=Y[7] *1.2:
                    A.append(7)
                elif k == 8 and min(M) <= Y[8]*1.2 :
                    A.append(8)
                elif k == 9 and min(M) <= Y[9]*1.2 :
                    A.append(9)
                elif k == 10 and min(M) <= Y[10]*1.2:
                    A.append(10)
                elif k == 11 and min(M) <= Y[11]*1.2:
                    A.append(11)
                elif k == 12 and min(M) <=Y[12]*1.2:
                    A.append(12)
                elif k == 13 and min(M) <=Y[13]*1.2:
                    A.append(13)
                elif k == 14 and min(M) <= Y[14]*1.2 :
                    A.append(14)
                elif k == 15 and min(M) <= Y[15]*1.2:
                    A.append(15)
                elif k == 16 and min(M) <=Y[16]*1.2:
                    A.append(16)
                elif k == 17 and min(M) <= Y[17]*1.2:
                    A.append(17)
                elif k == 18 and min(M) <=Y[18]*1.2 :
                    A.append(18)
                elif k == 19 and min(M) <=Y[19]*1.2:
                    A.append(19)

                elif k == 20 and min(M) <=Y[20]*1.2:
                    A.append(20)
                elif k == 21 and min(M) <= Y[21]*1.2:
                    A.append(21)
                elif k == 22 and min(M) <=Y[22]*1.2:
                    A.append(22)
                elif k == 23 and min(M) <=Y[23]*1.2:
                    A.append(23)
                elif k == 24 and min(M) <=Y[24]*1.2:
                    A.append(24)
                elif k == 25 and min(M) <=Y[25]*1.2:
                    A.append(25)
                elif k == 26 and min(M) <=Y[26]*1.2 :
                    A.append(26)
                elif k == 27 and min(M) <=Y[27]*1.2 :
                    A.append(27)
                elif k == 28 and min(M) <=Y[28]*1.2:
                    A.append(28)
                elif k == 29 and min(M) <=Y[29]*1.2:
                    A.append(29)
                elif k == 30 and min(M) <=Y[30]*1.2:
                    A.append(30)
                elif k == 31 and min(M) <=Y[31]*1.2:
                    A.append(31)
                else:
                    A.append(32)





dataframe = pd.DataFrame({'PID Clustering1':A})
dataframe.to_csv("test4.csv",index=False,sep=',')
