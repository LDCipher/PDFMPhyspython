import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class DataSet(object):
    def __init__(self):
        self.x = []
        self.Qs = []
        self.flavours = []
        self.F = []

    def split_data(self, Q_in: float, flav_in: float):
        # given Q value and falvour value give all x
        # flavour is defined by:
        flav_index = self.flavours.index(flav_in)
        Q_index = self.Qs.index(Q_in)

        x_values = []
        # loop over F
        for i in range(Q_index, len(self.F), len(self.Qs)):
            x_values.append(self.F[i][flav_index])
        
        # Convert x_values to a NumPy array for curve fitting
        x_values = np.array(x_values)

        # Fit the function to the data
        popt, _ = curve_fit(fit_function, self.x, x_values)

        # popt contains the fitted parameters: a, b, and c
        a, b, c = popt

        # Create x values for the fitted function
        fitted_x = np.linspace(0, 1, 100)
        fitted_y = fit_function(fitted_x, a, b, c)

        # Plot the original data and the fitted function
        plt.figure()
        plt.scatter(self.x, x_values, label='Original Data', color='b')
        plt.plot(fitted_x, fitted_y, label='Fitted Function', color='r')
        plt.xlabel('x')
        plt.ylabel('Function Value')
        plt.legend()
        plt.show()
        
        return a, b, c 


def fit_function(x, a, b, c):
    return a * (x ** b) * ((1 - x) ** c)

def get_array(line):
    temp = line.strip()
    temp = temp.split(' ')
    temp_out = []
    for i in range(0, len(temp)):
        temp_out.append(float(temp[i]))
    return temp_out


def process_file(file_path):
    All_x = []
    All_Qs = []
    x = []
    Qs = []
    ## if (line == '---')
    AllDatasets = []
    k = 0
    H = 0

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
                    tempDataset = DataSet()
                    j = 0
                if j == 1:
                    tempDataset.x = get_array(line)
                elif (j == 2):
                    tempDataset.Qs = get_array(line)
                elif (j == 3):
                    tempDataset.flavours = get_array(line)
                elif j > 3:
                    tempDataset.F.append(get_array(line))

            j += 1
            i += 1
    return AllDatasets


def main():
    file_path = 'MSHT20lo_as130_0000.dat'
    DataSets = []
    DataSets = process_file(file_path)

    print(f'x is : {DataSets[1].x}')
    print(f'Q is : {DataSets[1].Qs}')
    print(f'flavour is : {DataSets[1].flavours}')
    print(f'F is: {DataSets[1].F}')
    print(f'x_values: {DataSets[2].split_data(3.162278e+04, 1)}')

    # print F for given flavour and Q


main()
