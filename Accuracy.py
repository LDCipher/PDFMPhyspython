import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import h5py


class CLF(object):
    def __init__(self, x, y):
        data = []
        for counter in range(0, len(x)):
            data.append([x[counter], y[counter]])
        self.data = data
        self.newData = []

    def ADD(self, A, B):
        return [A[0] + B[0], A[1] + B[1]]

    def tSK(self, k, A):
        return [A[0] * k, A[1] * k]

    def Bt(self, t, P0, P1, P2):
        # P0, P1, P2 are vectors i.e. [x,y]
        A = self.ADD(self.tSK(1 - t, P0), self.tSK(t, P1))
        A = self.tSK(1 - t, A)

        B = self.ADD(self.tSK(1 - t, P1), self.tSK(t, P2))
        B = self.tSK(t, B)

        return self.ADD(A, B)

    def Quadratic_Berzier_curve(self, interval):
        # Generates a set of data for the whole data set
        data = []
        # print(1 / interval)

        for counter in range(1, len(self.data) - 1, 3):
            # print("HEY!")
            # print("counter: ",counter)
            P0 = self.data[counter - 1]
            P1 = self.data[counter]
            P2 = self.data[counter + 1]
            t = 0
            for i in range(0, int(1 / interval)):
                # singular point assignment
                self.newData.append(self.Bt(t, P0, P1, P2))
                t += interval

    def splitArray(self):
        x = []
        y = []
        for point in self.newData:
            x.append(point[0])
            y.append(point[1])
        return x, y

    def cubic_bezier_curve(self, t, P0, P1, P2, P3):
        # Explicit polynomial representation of cubic Bezier curve
        term1 = (1 - t) ** 3 * np.array(P0)
        term2 = 3 * (1 - t) ** 2 * t * np.array(P1)
        term3 = 3 * (1 - t) * t ** 2 * np.array(P2)
        term4 = t ** 3 * np.array(P3)
        return term1 + term2 + term3 + term4

    def cubic_bezier_fit(self, x_values, xf_values):
        # Fit cubic Bezier curve to the data
        popt, _ = curve_fit(self.cubic_bezier_curve, x_values, xf_values)
        return popt


def fit_function(x, a, b, c):
    return a * (x ** b) * ((1 - x) ** c)


def cubic_bezier_fit_function(x, af, bf, cf, p_0, p_1, p_2, p_3):
    f = (af * (x ** bf) * ((1 - x) ** cf)) * ((p_0 * ((-x ** 3) + (3 * x ** 2) - (3 * x) + 1)) + (
            p_1 * ((3 * x ** 3) - (6 * x ** 2) + (3 * x))) + (
                                                      p_2 * (-(3 * x ** 3) + (3 * x ** 2))) + (
                                                      p_3 * (x ** 3)))
    return f


def calculate_residuals_for_optimization(split_x, xf_values, x_values, error):
    return calculate_residuals(xf_values, x_values, split_x, error)


def fit_data(xf_values, x_values, split_x, error):
    xf_values = np.array(xf_values)
    x_values = np.array(x_values)

    # Split data into two regions based on the split_x value
    mask1 = x_values < split_x
    mask2 = x_values >= split_x

    # Perform separate curve fitting for each region
    result1 = curve_fit(fit_function, x_values[mask1], xf_values[mask1], sigma=error[mask1], absolute_sigma=True)
    popt1 = result1[0]

    result2 = curve_fit(fit_function, x_values[mask2], xf_values[mask2], sigma=error[mask2], absolute_sigma=True)
    popt2 = result2[0]
    return mask1, mask2, popt1, popt2


def calculate_residuals(xf_values, x_values, split_x, error):
    mask1, mask2, popt1, popt2 = fit_data(xf_values, x_values, split_x, error)

    residuals1 = xf_values[mask1] - fit_function(x_values[mask1], *popt1)
    residuals2 = xf_values[mask2] - fit_function(x_values[mask2], *popt2)

    total_residual = np.sum(np.concatenate((residuals1, residuals2)) ** 2)

    return total_residual


def find_best_split_x_optimized(xf_values, x_values, error):
    result = minimize_scalar(calculate_residuals_for_optimization, args=(xf_values, x_values, error),
                             bounds=(1e-5, 1 / 3), method='bounded')
    best_split_x = result.x
    return best_split_x


def plot_data(xf_values, x_values, split_x, error, min_err, max_err, q_value, flavour_value, x, y, bezier):
    mask1, mask2, popt1, popt2 = fit_data(xf_values, x_values, split_x, error)

    # Extract parameters for each region
    a1, b1, c1 = popt1
    a2, b2, c2 = popt2
    a3, b3, c3, d3, e3, f3, g3 = bezier
    # Create fitted curves for each region
    fitted_x1 = np.linspace(0, np.max(x_values[mask1]), 100000)
    fitted_x2 = np.linspace(np.min(x_values[mask2]), 1, 100000)
    fitted_x = np.linspace(0, np.max(x_values), 100000)
    fitted_y1 = fit_function(fitted_x1, a1, b1, c1)
    fitted_y2 = fit_function(fitted_x2, a2, b2, c2)
    fitted_y = cubic_bezier_fit_function(fitted_x, a3, b3, c3, d3, e3, f3, g3)
    plt.figure()
    plt.plot(x, y, label='Bezier Curve', color='y')
    plt.plot(fitted_x, fitted_y, label='Fitted Bezier', color='k')
    plt.plot(fitted_x1, fitted_y1, label='Fitted Function (Region 1)', color='r')
    plt.plot(fitted_x2, fitted_y2, label='Fitted Function (Region 2)', color='g')
    plt.errorbar(x_values, xf_values, yerr=[min_err, max_err], fmt='|', label='Original Data', color='b')
    plt.xlabel('x')
    plt.ylabel('Function Value')
    plt.xlim(np.amin(x_values), np.amax(x_values))
    plt.ylim(np.amin(xf_values), np.amax(xf_values))
    #plt.xscale('log')
    plt.title('Q value: ' + str(q_value) + '\nFlavour value: ' + str(flavour_value))
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
        min_err = np.array(q.get('min'))
        max_err = np.array(q.get('max'))
        return min_err, max_err


def main():
    filepath = 'c:/Users/LD_Ci/Documents/Pythonshit/PDFMPhyspython'
    q_value = 1.4
    flavour_value = 1

    xf_values = read_datasets_from_h5py(filepath + '/hp5y_data.h5', q_value, flavour_value)
    x_values = read_x_from_h5py(filepath + '/hp5y_data.h5')
    min_val, max_val = read_errors_from_h5py(filepath + '/hp5y_error.h5', q_value, flavour_value)
    min_err = xf_values - min_val
    max_err = max_val - xf_values
    error = np.maximum(min_err, max_err)
    split_x = find_best_split_x_optimized(xf_values, x_values, error)  # Define the split point
    a = CLF(x_values, xf_values)
    a.Quadratic_Berzier_curve(0.01)
    x, y = a.splitArray()
    var = [1] * len(x_values)
    popt, pcov = curve_fit(cubic_bezier_fit_function, x_values, xf_values, sigma=var)
    print("Optimal parameters are:", popt)
    params_region1, params_region2 = plot_data(xf_values, x_values, split_x, error, min_err, max_err,
                                               q_value, flavour_value, x, y, popt)
    # print(f'x is : {x_values}')
    # print(f'Q is : {DataSets[0].Qs}')
    # print(f'flavour is : {DataSets[0].flavours}')
    # print(f'F is: {xf_values}')
    print(f'low x parameters: {params_region1}')
    print(f'high x parameters: {params_region2}')
    print(f'Best split_x value: {split_x}')
    # print F for given flavour and Q


main()
