# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:23:35 2023

@author: LD_Ci
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class DataSet(object):
    def __init__(self):
        self.x = []
        self.Qs = []
        self.flavours = []
        self.F = []
        
    def getdata(self, in_flavours, in_Qs):
        index_flavours = self.flavours.index(in_flavours)
        index_Qs = self.Qs.index(in_Qs)
        return self.F[index_flavours][index_Qs]
    
def func(x, a, b, c):
    return a * (x ** b) * ((1 - x) ** c)
    
def get_array(line):
    temp = line.strip()
    temp = temp.split(' ')
    temp_out = []
    for i in range(0, len(temp)):
        temp_out.append(float(temp[i]))
    return temp_out


def process_file(file_path):
    ## if (line == '---')
    AllDatasets = []

    with open(file_path, 'r') as file:
        i = 0
        j = -1
        temp_F = []
        tempDataset = None
        specialLine = ''
        for line in file:
            # process each line
            if (i == 2):
                specialLine = line

            if (i in [0, 1]):
                print("ignore line")
            else:
                if (line == specialLine):
                    # make a dataset
                    print(line)
                    if tempDataset != None:
                        print("Update")
                        AllDatasets.append(tempDataset)
                        temp_F = []
                    tempDataset = DataSet()
                    j = 0
                if j == 1:
                    tempDataset.x = get_array(line)
                elif (j == 2):
                    tempDataset.Qs = get_array(line)
                elif (j == 3):
                    tempDataset.flavours = get_array(line)
                elif j > 3:
                    temp_F += get_array(line)
                    if ((j - 3) % (len(tempDataset.Qs)) == 0):
                        tempDataset.F.append(temp_F)
                        temp_F = []

            j += 1
            i += 1
    return AllDatasets


def main():
    file_path = 'MSHT20lo_as130_0000.dat'
    DataSets = []
    DataSets = process_file(file_path)
    a = DataSets[1]
    print(a)
    


main()
