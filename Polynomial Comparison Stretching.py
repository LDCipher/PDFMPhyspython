import numpy as np
from scipy.optimize import curve_fit
from scipy.special import binom
import matplotlib.pyplot as plt
import h5py
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL


def add(a, b):
    return [a[0] + b[0], a[1] + b[1]]


def tsk(k, a):
    return [a[0] * k, a[1] * k]


def bt(t, p_0, p_1, p_2):
    # P0, P1, P2 are vectors i.e. [x,y]
    a = add(tsk(1 - t, p_0), tsk(t, p_1))
    a = tsk(1 - t, a)

    b = add(tsk(1 - t, p_1), tsk(t, p_2))
    b = tsk(t, b)

    return add(a, b)


class CLF(object):
    def __init__(self, x, y):
        data = []
        for counter in range(0, len(x)):
            data.append([x[counter], y[counter]])
        self.data = data
        self.newData = []

    def quadratic_bezier_curve(self, interval):
        # print(1 / interval)

        for counter in range(1, len(self.data) - 1, 3):
            # print("HEY!")
            # print("counter: ",counter)
            p0 = self.data[counter - 1]
            p1 = self.data[counter]
            p2 = self.data[counter + 1]
            t = 0
            for i in range(0, int(1 / interval)):
                # singular point assignment
                self.newData.append(bt(t, p0, p1, p2))
                t += interval

    def split_array(self):
        x = []
        y = []
        for point in self.newData:
            x.append(point[0])
            y.append(point[1])
        return x, y


def bezier_fit_function(x, *args):
    temp = np.zeros(len(x))
    af, bf, cf, alpha = args[0], args[1], args[2], args[3]
    p_arg = list(args[4:])
    p_arg.append(0)
    n = len(p_arg) - 1
    y = np.power(x, alpha)
    for i, p in enumerate(p_arg):
        temp = np.add((p * bernstein_basis_function(y, i, n)), temp)

    f = (af * (x ** bf) * ((1 - x) ** cf)) * temp
    return f


def bernstein_basis_function(y, i, n):
    return (binom(n, i)) * (y ** i) * ((1 - y) ** (n - i))


def calculate_bezier(x_values, xf_values, interval):
    bezier = CLF(x_values, xf_values)
    bezier.quadratic_bezier_curve(interval)
    bezier_x, bezier_y = bezier.split_array()
    return np.array(bezier_y), np.array(bezier_x)


def plot_data(xf_values, x_values, min_val, max_val, q_value, flavour_value, interval, n):
    # Get interpolated values
    y, x = calculate_bezier(x_values, xf_values, interval)
    bez_err_max, _ = calculate_bezier(x_values, max_val, interval)
    bez_err_min, _ = calculate_bezier(x_values, min_val, interval)

    # Remove small values
    mask = x < 1

    # Calculate errors
    min_err = xf_values - min_val
    max_err = max_val - xf_values

    # Fit bezier function to curve
    temp = np.array(np.zeros(int(n + 4)))
    temp[[1, 2, 3]] = [-0.5, 0.5, 0.5]
    bounds_0 = np.array(np.ones(int(n + 4)) * -1 * np.inf)
    bounds_0[[0, 2, 3]] = [0, np.negative(n), 0]
    bounds_1 = np.array(np.ones(int(n + 4)) * np.inf)
    bounds_1[[1]] = [0]
    bounds = [bounds_0, bounds_1]
    popt, _ = curve_fit(bezier_fit_function, x[mask], y[mask], p0=temp, bounds=bounds)

    # Fitted data
    fitted_x = np.array(x)
    fitted_y = bezier_fit_function(fitted_x, *popt)

    # Plot data

    plt.figure()
    plt.plot(x, y, label='Bezier Curve', color='y')
    plt.plot(fitted_x, fitted_y, label='Fitted Bezier', color='r')
    plt.errorbar(x_values, xf_values, yerr=[min_err, max_err], fmt='|', label='Data', color='b')
    plt.xlabel('x')
    plt.ylabel('xf(x,Q²)')
    plt.xlim(np.amin(x_values), np.amax(x_values))
    plt.ylim(np.amin(xf_values), np.amax(xf_values))
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 1)
    plt.xscale('log')
    plt.title('Q value: ' + str(q_value) + ' GeV\nFlavour value: down quark')
    plt.legend()
    plt.show()

    # Percentage data
    err_fit = (np.abs(np.array(fitted_y) / np.array(y))) * 100
    err_max = (np.array(bez_err_max) / np.array(y)) * 100
    err_min = (np.array(bez_err_min) / np.array(y)) * 100
    y = (np.array(y) / np.array(y)) * 100

    # Plot errors
    plt.figure()
    plt.plot(x, (y - y), label='Bezier Curve', color='y')
    plt.plot(fitted_x, (err_fit - y), label='Fitted Bezier', color='r')
    plt.plot(x, (err_max - y), label='Upper Bound', color='b')
    plt.plot(x, (err_min - y), label='Lower Bound', color='b')
    plt.xlabel('x')
    plt.ylabel('Deviation from interpolated data (%)')
    plt.xlim(np.amin(x_values), np.amax(x_values))
    plt.ylim(-10, 10)
    plt.xscale('log')
    plt.title('Q value: ' + str(q_value) + ' GeV\nFlavour value: down quark')
    plt.legend()
    plt.show()

    return popt, (x, y)


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
    flavour_value = 2
    bezier_interval = 0.01

    xf_values = read_datasets_from_h5py(filepath + '/hp5y_data.h5', q_value, flavour_value)
    x_values = read_x_from_h5py(filepath + '/hp5y_data.h5')
    min_val, max_val = read_errors_from_h5py(filepath + '/hp5y_error.h5', q_value, flavour_value)
    params, bezier_curve = plot_data(xf_values, x_values, min_val, max_val,
                                     q_value, flavour_value, bezier_interval, 6)

    print("Optimal parameters are:", params)


main()
