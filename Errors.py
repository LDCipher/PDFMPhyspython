import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import h5py


def fit_function(x, a, b, c):
    return a * (x ** b) * ((1 - x) ** c)


def calculate_residuals_for_optimization(split_x, xf_values, x_values):
    return calculate_residuals(xf_values, x_values, split_x)


def calculate_residuals(xf_values, x_values, split_x):
    xf_values = np.array(xf_values)
    x_values = np.array(x_values)

    mask1 = x_values < split_x
    mask2 = x_values >= split_x
    sigma1 = np.ones(len(x_values[mask1]))
    sigma1[[0, -1]] = 1
    sigma2 = np.ones(len(x_values[mask2]))
    sigma2[[0, -1]] = 1

    result1 = curve_fit(fit_function, x_values[mask1], xf_values[mask1], sigma=sigma1)
    popt1 = result1[0]

    result2 = curve_fit(fit_function, x_values[mask2], xf_values[mask2], sigma=sigma2)
    popt2 = result2[0]

    residuals1 = xf_values[mask1] - fit_function(x_values[mask1], *popt1)
    residuals2 = xf_values[mask2] - fit_function(x_values[mask2], *popt2)

    total_residual = np.sum(np.concatenate((residuals1, residuals2)) ** 2)

    return total_residual


def find_best_split_x_optimized(xf_values, x_values):
    result = minimize_scalar(calculate_residuals_for_optimization, args=(xf_values, x_values), bounds=(1e-5, 1 / 3),
                             method='bounded')
    best_split_x = result.x
    return best_split_x


def fit_data(xf_values, x_values, split_x):
    xf_values = np.array(xf_values)
    x_values = np.array(x_values)

    # Split data into two regions based on the split_x value
    mask1 = x_values < split_x
    mask2 = x_values >= split_x
    sigma1 = np.ones(len(x_values[mask1]))
    sigma1[[0, -1]] = 1
    sigma2 = np.ones(len(x_values[mask2]))
    sigma2[[0, -1]] = 1
    # print(xf_values[mask1])
    # Perform separate curve fitting for each region
    result1 = curve_fit(fit_function, x_values[mask1], xf_values[mask1], sigma=sigma1)
    popt1 = result1[0]

    result2 = curve_fit(fit_function, x_values[mask2], xf_values[mask2], sigma=sigma2)
    popt2 = result2[0]

    # Extract parameters for each region
    a1, b1, c1 = popt1
    a2, b2, c2 = popt2
    # Create fitted curves for each region
    fitted_x1 = np.linspace(0, np.max(x_values[mask1]), 100000)
    fitted_x2 = np.linspace(np.min(x_values[mask2]), 1, 100000)
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
    # plt.ylim(0, 10)
    plt.xscale('log')
    plt.legend()
    plt.show()

    return (a1, b1, c1), (a2, b2, c2)


def read_datasets_from_h5py(file_path, q_value, flv_value):
    with h5py.File(file_path, 'r') as hdf_file:
        flavour = hdf_file.get(str(float(flv_value)))
        q = flavour.get(str(float(q_value)))
        return np.array(q)


def read_x_from_h5py(file_path):
    with h5py.File(file_path, 'r') as hdf_file:
        x = hdf_file.get('x_values')
        return np.array(x)


def read_errors_from_h5py(file_path, q_value, flv_value):
    with h5py.File(file_path, 'r') as hdf_file:
        flavour = hdf_file.get(str(float(flv_value)))
        q = flavour.get(str(float(q_value)))
        return np.array(q)


def main():
    filepath = 'c:/Users/LD_Ci/Documents/Pythonshit/PDFMPhyspython'
    q_value = 1.4
    flavour_value = 2

    xf_values = read_datasets_from_h5py(filepath+'/hp5y_data.h5', q_value, flavour_value)
    x_values = read_x_from_h5py(filepath+'/hp5y_data.h5')
    error_values = read_errors_from_h5py(filepath+'/hp5y_error.h5')
    split_x = find_best_split_x_optimized(xf_values, x_values)  # Define the split point
    params_region1, params_region2 = fit_data(xf_values, x_values, split_x)
    # print(f'x is : {x_values}')
    # print(f'Q is : {DataSets[0].Qs}')
    # print(f'flavour is : {DataSets[0].flavours}')
    # print(f'F is: {DataSets[0].F}')
    print(f'low x parameters: {params_region1}')
    print(f'high x parameters: {params_region2}')
    print(f'Best split_x value: {split_x}')
    # print F for given flavour and Q


main()
