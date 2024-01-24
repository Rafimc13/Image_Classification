import time
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
import warnings


def load_data(file, column_name):
    my_file = sio.loadmat(file)
    if column_name is not None:
        data = my_file[column_name]
        return data
    else:
        return file


def plot_data(dataset, ylabel, xlabel, title):
    plt.figure(figsize=(10, 6))
    plt.plot(dataset)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def image_show(image, band):
    plt.figure(figsize=(10, 6))
    plt.imshow(image[:,:,band])
    plt.title(f'RGB Visualization of the {band}th band of Pavia University HSI')
    plt.show()


def calc_theta_LS(X, y):
    """
    Calculation of theta via Least Squares method.
        No constraints inserted
        """
    XtX = np.dot(np.transpose(X), X)
    XtX_inverse = np.linalg.inv(XtX)
    XtY = np.dot(np.transpose(X), y)
    my_theta = np.dot(XtX_inverse, XtY)

    return my_theta


def objective_function(theta, X, y):
    # Normalize the coefficients to ensure sum-to-one constraint
    theta_normalized = theta / np.sum(theta)
    return np.sum((X @ theta_normalized - y) ** 2)


def calc_theta_LS_constraints(X, y, bounds=False):
    num_of_theta = np.ones(9) / 9
    if bounds:
        bounds = [(0, None)] * 9
        result = minimize(objective_function, num_of_theta, args=(X, y), bounds=bounds)
    else:
        result = minimize(objective_function, num_of_theta, args=(X, y))

    my_theta = result.x / np.sum(result.x)
    return my_theta

def calc_theta_Lasso(X, y, lamda):
    """Calculation of theta. In order to use our function for multiple
        methods we add lamda as a fixed parameter."""
    XtX = np.dot(np.transpose(X), X)
    I_matrix = np.eye(XtX.shape[0])
    XtX_l = XtX + lamda*I_matrix
    XtX_l_inverse = np.linalg.inv(XtX_l)
    XtY = np.dot(np.transpose(X), y)
    my_theta = np.dot(XtX_l_inverse, XtY)

    return my_theta


def calc_MSE(thetas, X, y):
    MSE = 0
    for i in range(len(thetas)):
        mse = (y[i] - np.dot(X, thetas[i]))
        MSE += np.linalg.norm(mse)

    MSE = MSE/len(thetas)

    return MSE


def calc_abundance_map(thetas, endmembers, HSI, groundtruth, num_of_class=0):
    step = 0
    for i in range(len(HSI)):
        for j in range(len(HSI[i])):
            if groundtruth[i][j] != 0:
                HSI[i][j] = thetas[step][num_of_class] * endmembers[:, num_of_class]
                step += 1

    return HSI



image_file = 'data/PaviaU_cube.mat'  # Pavia HSI: 300x200x103
endmembers_file = 'data/PaviaU_endmembers.mat'  # Endmember's matrix: 103x9
ground_truth_file = 'data/PaviaU_ground_truth.mat'

HSI = load_data(image_file, 'X')
# print(len(HSI[2][3]))

endmembers = load_data(endmembers_file, 'endmembers')

# ylabel = 'Radiance values'
# xlabel = 'Spectral bands'
# title = '9 Endmembers spectral signatures of Pavia University HSI'
# my_data.plot_data(endmembers, ylabel, xlabel, title)
# my_data.image_show(HSI, 10)

ground_truth = load_data(ground_truth_file, 'y')
sum_zero = 0

non_zero_pixels = []
for i in range(len(ground_truth)):
    for j in range(len(ground_truth[i])):
        if ground_truth[i][j] == 0:
            sum_zero += 1
        else:
            non_zero_pixels.append(HSI[i][j])


non_zero_pixels = np.array(non_zero_pixels)
LS_thetas = []
for pixel in non_zero_pixels:
    LS_thetas.append(calc_theta_LS(endmembers, pixel))

print(LS_thetas[1])
MSE_LS = calc_MSE(LS_thetas, endmembers, non_zero_pixels)
print(MSE_LS)


for i in range(9):
    HSI_LS = calc_abundance_map(LS_thetas, endmembers, HSI, ground_truth, num_of_class=i)
    print(f'Plotting the abundance map for class/endmember: {i+1}')
    image_show(HSI_LS, 10)

# LS_thetas_sum_to_one = []
# for pixel in non_zero_pixels:
#     LS_thetas_sum_to_one.append(calc_theta_LS_constraints(endmembers, pixel))
# MSE_sum_to_one = calc_MSE(LS_thetas_sum_to_one, endmembers, non_zero_pixels)
# print(MSE_sum_to_one)


LS_thetas_non_negative = []
for pixel in non_zero_pixels:
    optimal, MSE_non_negative = nnls(endmembers, pixel)
    LS_thetas_non_negative.append(optimal)

print(MSE_non_negative)


# LS_thetas_constraints = []
# for pixel in non_zero_pixels:
#     LS_thetas_constraints.append(calc_theta_LS_constraints(endmembers, pixel, bounds=True))

# MSE_constraints = calc_MSE(LS_thetas_constraints, endmembers, non_zero_pixels)
# print(MSE_constraints)


# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=ConvergenceWarning)
#     l = 0.0
#     comp_MSE_lasso = 10000000
#     while l<=100:
#         l += 0.1
#         thetas_lasso = []
#         lasso_model = Lasso(alpha=l)
#         for pixel in non_zero_pixels:
#             lasso_model.fit(endmembers, pixel)
#             thetas_lasso.append(lasso_model.coef_)
#         MSE_lasso = calc_MSE(thetas_lasso, endmembers, non_zero_pixels)
#         if MSE_lasso <= comp_MSE_lasso:
#             best_MSE_lasso = {}
#             best_MSE_lasso[l] = MSE_lasso

# for key, value in best_MSE_lasso.items():
#     print(f"Best hyperparameter λ for the Lasso method is: {key} with MSE score: {value}")


