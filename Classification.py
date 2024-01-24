import scipy.io as sio
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def load_data(file, column_name):
    my_file = sio.loadmat(file)
    if column_name is not None:
        data = my_file[column_name]
        return data
    else:
        return file

def create_mean(set_y, set_x, num_class=9):
    """
    Estimate mean of a multivariate
    Gaussian distribution
    """
    num_dimensions = len(set_x[0])
    mean = np.zeros(num_dimensions)
    num_of_values = 0
    for i in range(len(set_y)):
        if set_y[i] == num_class:
            mean += set_x[i]
            num_of_values += 1

    if num_of_values>0:
        mean = np.array(mean / num_of_values)
        prob_w_class = num_of_values/len(set_y)
        return mean, prob_w_class
    else:
        print('No values assigned in this class')
        return None


def create_cov(y, X, mean, num_class=9):
    """
    Estimate covariance of a multivariate
    Gaussian distribution
    """
    num_dimensions = len(X[0])
    cov = np.zeros((num_dimensions, num_dimensions))
    num_of_values = 0
    for i in range(len(y)):
        if y[i] == num_class:
            cov += np.outer((X[i] - mean), (X[i] - mean))
            num_of_values += 1
    if num_of_values>0:
        cov = cov / num_of_values
        return cov
    else:
        print('No values assigned in this class')
        return None


def estimate_values(X, probs_w, means, covariances):
    """
    Calculate the probability of a new point x via
    the estimated pdf p(x/wj)
    """
    prob_y = []
    pdf_of_classes = []
    for i in range(len(means)):
        if len(means[i]) == 2 and len(covariances[i]) == 2:
            pdf_of_classes.append(None)
        else:
            my_pdf = multivariate_normal(mean=means[i], cov=covariances[i])
            pdf_of_classes.append(my_pdf)
    for i in range(len(X)):
        max = 0
        best_class = 0
        for j in range(len(pdf_of_classes)):
            if pdf_of_classes[j] is None:
                prob_x = 0
            else:
                prob_x = pdf_of_classes[j].pdf(X[i]) * probs_w[j]
            if prob_x >= max:
                max = prob_x
                best_class = j + 1
        prob_y.append(best_class)

    return prob_y


def calc_min_euclidean(X, means):
    prob_y = []
    for i in range(len(X)):
        min = 100000
        best_class = 0
        for j in range(len(means)):
            eucl_dist = np.sqrt(np.sum((X - means[j]) ** 2))
            if eucl_dist <= min:
                min = eucl_dist
                best_class = j+1
        prob_y.append(best_class)

    return prob_y


def train_and_evaluate(model_name, X, y, Xtest, k_fold=10 , num_classes=9):
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    accuracy_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        class_means = []
        class_covariances = []
        class_probs_w = []
        for class_number in range(1, num_classes+1):
            mean, prob_w = create_mean(y_train, X_train, class_number)
            cov = create_cov(y_train, X_train, mean, class_number)
            class_probs_w.append(prob_w)
            class_means.append(mean)
            class_covariances.append(cov)
        if model_name == "Bayes Classifier":
            predictions = estimate_values(X_test, class_probs_w, class_means, class_covariances)
            accuracy = accuracy_score(y_test, predictions)
            accuracy_scores.append(accuracy)
        else:
            predictions = calc_min_euclidean(X_test, class_means)
            accuracy = accuracy_score(y_test, predictions)
            accuracy_scores.append(accuracy)

    avg_accuracy = np.mean(accuracy_scores)
    std = np.std(accuracy_scores)
    print(f"Mean Validation Error of {model_name} model is: {(1-avg_accuracy)*100: .3f}%")
    print(f"Standard Deviation error of Bayes classifier model is: {std: .3f}")
    print("")

    # Train it in the whole training set
    class_means_all = []
    class_covariances_all = []
    class_probs_w_all = []
    for class_number in range(1, num_classes + 1):
        mean, prob_w = create_mean(y, X, class_number)
        cov = create_cov(y, X, mean, class_number)
        class_probs_w_all.append(prob_w)
        class_means_all.append(mean)
        class_covariances_all.append(cov)
    if model_name == "Bayes Classifier":
        y_pred_all = estimate_values(Xtest, class_probs_w_all, class_means_all, class_covariances_all)
    else:
        y_pred_all = calc_min_euclidean(Xtest, class_means_all)

    return y_pred_all

def split_training_set(X, y, Xtest):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j] != 0:
                X_train.append(X[i][j])
                y_train.append(y[i][j])
            elif Xtest[i][j] != 0:
                X_test.append(X[i][j])
                y_test.append(Xtest[i][j])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def model_classify(model, model_name, n_splits, X_train, y_train, X_test):
    my_model = model
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Make predictions on the test data
    cv_results = cross_val_score(my_model, X_train, y_train, cv=kf)
    mean = cv_results.mean()
    std = cv_results.std()
    print(f"Mean Validation Error of {model_name} model is: {(1 -mean)*100: .3f}%")
    print(f"Standard Deviation error of {model_name} model is: {std: .3f}")
    print("")
    # Train the classifier on the whole training data
    my_model.fit(X_train, y_train)
    y_pred_all = my_model.predict(X_test)
    return y_pred_all


def calc_confusion_matrix (model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(f"The confusion matrix for {model_name} model is: {cm}")
    diagonal_sum = np.sum(np.diagonal(cm))
    sum_cm = np.sum(cm)
    accuracy = diagonal_sum/sum_cm
    print(f"The success rate of {model_name} model is: {accuracy*100: .3f}%")
    print("")
    return accuracy

image_file = 'data/PaviaU_cube.mat'  # Pavia HSI: 300x200x103
endmembers_file = 'data/PaviaU_endmembers.mat'  # Endmember's matrix: 103x9
ground_truth_file = 'data/PaviaU_ground_truth.mat'

HSI = load_data(image_file, 'X')
endmembers = load_data(endmembers_file, 'endmembers')
ground_truth = load_data(ground_truth_file, 'y')

# Training set for classification
Pavia_labels = sio.loadmat('data/classification_labels_Pavia.mat')
Training_Set = (np.reshape(Pavia_labels['training_set'],(200,300))).T
Test_Set = (np.reshape(Pavia_labels['test_set'],(200,300))).T
Operational_Set = (np.reshape(Pavia_labels['operational_set'],(200,300))).T


nb_model = GaussianNB()
knn_model = KNeighborsClassifier()

X_train, y_train, X_test, y_test = split_training_set(HSI, Training_Set, Test_Set)

nb_preds = model_classify(nb_model, 'Naive Bayes', 10, X_train, y_train, X_test)
knn_preds = model_classify(knn_model, 'k-nearest neighbor', 10,  X_train, y_train, X_test)




# Compute the confusion matrix and success rate for Naive Bayes
nb_accuracy = calc_confusion_matrix('Naive Bayes', y_test, nb_preds)
# Compute the confusion matrix and success rate for k-nearest neighbor
knn_accuracy = calc_confusion_matrix('k-nearest neighbor', y_test, knn_preds)

# Compute the confusion matrix and success rate for Bayesian Classifier
bayes_preds = train_and_evaluate('Bayes Classifier', X_train, y_train, X_test)
bayes_accuracy = calc_confusion_matrix('Bayes Classifier', y_test, bayes_preds)


euclidean_preds = train_and_evaluate('minimun Euclidean distance classifier', X_train, y_train, X_test)
euclidean_accuracy = calc_confusion_matrix('minimun Euclidean distance classifier', y_test, euclidean_preds)