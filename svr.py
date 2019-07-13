#!/usr/local/bin/python
import csv
from sklearn import svm

fname = "svr"

def svr(train,test):
    f = open(train, 'rt')
    data_rows = csv.reader(f)

    x_values = []
    y_values = []
    for i,row in enumerate(data_rows):
        if i == 0:
            continue
        for j,cell in enumerate(row):
            if j != 0:
                row[j] = float(cell)
        x_values.append(row[2:])
        y_values.append(row[1])
    clf = svm.SVR()
    clf.fit(x_values,y_values)
    f.close()

    f = open(test, 'rt')
    data_rows = csv.reader(f)
    x_values = []
    y_values = []
    for i,row in enumerate(data_rows):
        if i == 0:
            continue
        for j, cell in enumerate(row):
            if j != 0:
                row[j] = float(cell)
        x_values.append(row[1:])
        y_values.append(row[0])
    f.close()
    results = clf.predict(x_values)

    f = open(fname+"_out.csv", 'w')
    fwrite = csv.writer(f, delimiter=',')
    fwrite.writerow(['Patient', 'Target', 'Alzheimers Chance'])
    for i,row in enumerate(results):
        chance = 'Positive' if row < 25 else 'Negetive'
        fwrite.writerow([y_values[i],row, chance])
    f.close()

if __name__ == '__main__':
    svr('fMRI_train.csv', 'fMRI_test.csv')