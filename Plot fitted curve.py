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

        xf_values = []

        for i in range(Q_index, len(self.F), len(self.Qs)):
            xf_values.append(self.F[i][flav_index])
        return xf_values

def fit_function(x, a, b, c):
    epsilon = 1e-20
    x = np.maximum(x, epsilon)
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

def calculate_residuals(xf_values, x_values, split_x):
    xf_values = np.array(xf_values)
    x_values = np.array(x_values)

 
    mask1 = x_values < split_x
    mask2 = x_values >= split_x

    popt1, _ = curve_fit(fit_function, x_values[mask1], xf_values[mask1])
    try: 
        popt2, _ = curve_fit(fit_function, x_values[mask2], xf_values[mask2])

    except:
        return np.inf
    residuals1 = xf_values[mask1] - fit_function(x_values[mask1], *popt1)
    residuals2 = xf_values[mask2] - fit_function(x_values[mask2], *popt2)

 
    total_residual = np.sum(np.concatenate((residuals1, residuals2))**2)

    return total_residual

def find_best_split_x(xf_values, x_values):
 
    split_x_values = np.linspace(4e-4, 0.4, 1000)

    best_split_x = None
    best_residual = np.inf

    for split_x in split_x_values:
        residual = calculate_residuals(xf_values, x_values, split_x)
        if residual < best_residual:
            best_residual = residual
            best_split_x = split_x

    return best_split_x

def fit_data(xf_values, x_values, split_x):
    xf_values = np.array(xf_values)
    x_values = np.array(x_values)

    # Split data into two regions based on the split_x value
    mask1 = x_values < split_x
    mask2 = x_values >= split_x
    #print(xf_values[mask1])
    # Perform separate curve fitting for each region
    popt1, _ = curve_fit(fit_function, x_values[mask1], xf_values[mask1])
    popt2, _ = curve_fit(fit_function, x_values[mask2], xf_values[mask2])

    # Extract parameters for each region
    a1, b1, c1 = popt1
    a2, b2, c2 = popt2

    # Create fitted curves for each region
    fitted_x1 = np.linspace(0, split_x, 100000)
    fitted_x2 = np.linspace(split_x, 1, 100000)
    fitted_y1 = fit_function(fitted_x1, a1, b1, c1)
    fitted_y2 = fit_function(fitted_x2, a2, b2, c2)

    plt.figure()
    plt.scatter(x_values, xf_values, label='Original Data', color='b')
    plt.plot(fitted_x1, fitted_y1, label='Fitted Function (Region 1)', color='r')
    plt.plot(fitted_x2, fitted_y2, label='Fitted Function (Region 2)', color='g')
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.xlim(np.amin(x_values), np.amax(x_values))
    plt.ylim(np.amin(xf_values), np.amax(xf_values))
    plt.xscale('log')
    plt.legend()
    plt.show()

    return (a1, b1, c1), (a2, b2, c2)


def main():
    file_path = 'MSHT20lo_as130_0000.dat'
    DataSets = []
    DataSets = process_file(file_path)
    xf_values = DataSets[0].split_data(1.4, 1)
    x_values = DataSets[0].x
    split_x = find_best_split_x(xf_values, x_values)  # Define the split point
    #split_x = 0.0001
    params_region1, params_region2 = fit_data(xf_values, x_values, split_x)
    #fit_data(np.abs(DataSets[0].split_data(1, 1)[40:]), DataSets[0].x[40:])
    #print(f'x is : {x_values}')
    #print(f'Q is : {DataSets[0].Qs}')
    #print(f'flavour is : {DataSets[0].flavours}')
    #print(f'F is: {DataSets[0].F}')
    print(f'low x parameters: {params_region1}')
    print(f'high x parameters: {params_region2}')
    print(f'Best split_x value: {split_x}')

    # print F for given flavour and Q


main()
