import csv
import re
import pandas as pd
from numpy import *
import numpy
train = pd.read_csv('D:\python file\paper\Clustering\TEXT4-1.csv',delimiter=',')
df = pd.DataFrame(train)
trian1=df.replace('?', -1)
#numpy.savetxt('TEXT1.csv',trian1, delimiter=',')
name=['setpoint','gain','reset rate','deadband','cycle time','rate','system mode','control scheme','pump','solenoid','pressure measurement']
dataframe = pd.DataFrame(columns=name,data=trian1)
dataframe.to_csv("TEXT4-2.csv",index=False,sep=',')

#'setpoint','gain','reset rate','deadband','cycle time','rate','system mode','control scheme','pump','solenoid','pressure measurement'