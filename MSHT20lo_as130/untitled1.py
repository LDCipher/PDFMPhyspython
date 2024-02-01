# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:23:35 2023

@author: LD_Ci
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the function you want to fit
def func(x, a, b, c):
    return a * (x ** b) * ((1 - x) ** c)

# Read data from a text file with spaces as delimiters, within a specified range of rows and a single column
def read_data_from_file(file_path, column, start_row, end_row):
    data = np.loadtxt(file_path, usecols=(column - 1,), skiprows=12)  # Read only the specified column
    data_range = data[start_row-1:end_row]
    return data_range

# Input file path
input_file = 'test.csv'  # Replace with your file path
column_to_read = 1  # Replace with the desired column number
start_row = 0  # Replace with the starting row number
end_row = 500  # Replace with the ending row number

# Read the data from the specified range of rows and the specified column in the input file
data_range = read_data_from_file(input_file, column_to_read, start_row, end_row)

# Create x_data as an array of sequential numbers based on the length of data_range
x_data = np.arange(1, len(data_range) + 1)

# Use data_range as y_data
y_data = data_range

# Fit the function to the data
params, covariance = curve_fit(func, x_data, y_data)

# Extract the fitted parameters
a_fit, b_fit, c_fit = params

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Data')
x_fit = np.linspace(1, len(data_range), 100)
y_fit = func(x_fit, a_fit, b_fit, c_fit)
plt.plot(x_fit, y_fit, 'r', label='Fitted Curve')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fitting a Function with 3 Parameters')
plt.show()

# Print the fitted parameters
print("Fitted Parameters:")
print("a:", a_fit)
print("b:", b_fit)
print("c:", c_fit)

