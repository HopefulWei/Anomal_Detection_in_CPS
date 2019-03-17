import csv

data=[]
i = 0
j = 0

with open('data.txt','r') as f:
    for line in f.readlines():
        m = line.split(',')
        data = list(m)

with open('data1.txt','r') as f1:
    for line in f1.readlines():
        data1 = line
        print(data1)



with open('TEXT1.csv',"rt") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a = float(row[0])
        if data[i] == ' 1'and a == 1:
            j = j + 1
        i = i + 1

print(j)
b = int(data1)
c = j/b
print(c)

