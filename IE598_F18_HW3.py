#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:27:11 2018

@author: kirktsui
"""


from urllib.request import urlopen
import numpy as np

#read data from uci data repository
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urlopen(target_url)

#arrange data into list for labels and list of lists for attributes
xList = []
labels = []

for line in data:
#split on comma
#    row = line.decode('utf-8').strip().split(",")
    row = line.strip().split(str.encode(","))
    xList.append(row)

print ("Number of Rows of Data = " + str(len(xList)) + '\n')
print ("Number of Columns of Data = " + str(len(xList[1])))

 
nrow = len(xList)
ncol = len(xList[1])
type = [0]*3
colCounts = []
for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
print ("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    print(str(iCol) + '\t' + str(types[0]) + '\t' + str(types[1]) + '\t' + str(types[2]) + "\n")
    iCol += 1

#generate summary statistics for column 3 (e.g.)
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))
    colArray = np.array(colData) 
    colMean = np.mean(colArray)
    colsd = np.std(colArray)
print ("Mean = " + str(colMean) + '\t' +"Standard Deviation = "+ str(colsd) + "\n")
#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
print("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
print("\n")
#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
    
print("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
print(" \n")
#The last column contains categorical variables
col = 60
colData = []
for row in xList:
    colData.append(row[col])
    
unique = set(colData)
print("Unique Label Values \n") 
print(unique)
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*2
for elt in colData:
    catCount[catDict[elt]] += 1 
print("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount) 

#generate summary statistics for column 3 (e.g.) 
import pylab
import scipy.stats as stats
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))

stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()


#pandas

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
 "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
 #read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(target_url,header=None, prefix="V")
#print head and tail of data frame
print(rocksVMines.head())
print(rocksVMines.tail())
#print summary of data frame
summary = rocksVMines.describe()
print(summary) 
 

for i in range(208):
    #assign color based on "M" or "R" labels
    if rocksVMines.iat[i,60] == "M":
        pcolor = "red"
    else:
        pcolor = "blue"
#plot rows of data as if they were series data
    dataRow = rocksVMines.iloc[i,0:60]
    dataRow.plot(color=pcolor)
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Values")
plot.show()


#calculate correlations between real-valued attributes
dataCol2 = rocksVMines.iloc[:,1]
dataCol3 = rocksVMines.iloc[:,2]
plot.scatter(dataCol2, dataCol3)
plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()

dataCol21 = rocksVMines.iloc[:,20]

plot.scatter(dataCol2, dataCol21)

plot.xlabel("2nd Attribute")
plot.ylabel(("21st Attribute"))
plot.show()
 

from random import uniform

target = []
for i in range(208):
 #assign 0 or 1 target value based on "M" or "R" labels
     if rocksVMines.iat[i,60] == "M":
         target.append(1.0)
     else:
         target.append(0.0)
     #plot 35th attribute
dataCol = rocksVMines.iloc[0:208,35]
plot.scatter(dataCol, target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()
 #
 #To improve the visualization, this version dithers the points a little
 # and makes them somewhat transparent

target = [] #target needs to be reinitialized
for i in range(208):
 #assign 0 or 1 target value based on "M" or "R" labels
     # and add some dither
     if rocksVMines.iat[i,60] == "M":
         target.append(1.0 + uniform(-0.1, 0.1))
     else:
         target.append(0.0 + uniform(-0.1, 0.1))
#plot 35th attribute with semi-opaque points
dataCol = rocksVMines.iloc[0:208,35]
plot.scatter(dataCol, target, alpha=0.5, s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

 

from math import sqrt 
 
#calculate correlations between real-valued attributes
dataCol2 = rocksVMines.iloc[1,0:60]
dataCol3 = rocksVMines.iloc[2,0:60]
dataCol21 = rocksVMines.iloc[20,0:60]
mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataCol2)
for i in range(numElt):
    mean2 += dataCol2[i]/numElt
    mean3 += dataCol3[i]/numElt
    mean21 += dataCol21[i]/numElt
var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataCol2[i] - mean2) * (dataCol2[i] - mean2)/numElt
    var3 += (dataCol3[i] - mean3) * (dataCol3[i] - mean3)/numElt
    var21 += (dataCol21[i] - mean21) * (dataCol21[i] - mean21)/numElt
corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (dataCol2[i] - mean2) * \
    (dataCol3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr221 += (dataCol2[i] - mean2) * \
    (dataCol21[i] - mean21) / (sqrt(var2*var21) * numElt)
print("Correlation between attribute 2 and 3 \n") 
print(corr23)
print(" \n")
print("Correlation between attribute 2 and 21 \n")
print(corr221)
print(" \n")
 
 
#calculate correlations between real-valued attributes
corMat = DataFrame(rocksVMines.corr())
#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()

print("My name is Jianhao Cui")
print("My NetID is: jianhao3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
