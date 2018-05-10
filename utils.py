import numpy as np


'''
This file contains all the helper methods
'''


def error(pred, test):
    '''
    Average percentage error, use this for time series
    '''

    perc_err = 0
    for i in range(len(pred)):
        perc_err += (abs(pred[i] - test[i][0]) / test[i][0] * 100)
    print(perc_err / len(test))

def EWMA(train, test, alpha):
    pred = [train[-1][0]]
    for i in range(test.shape[0] - 1):
        n_pred = alpha * test[i][0] + (1 - alpha) * pred[i]
        pred.append(n_pred)
    return pred


def Seasonal(train, test, s):
    pred = []
    for i in range(len(test)):
        pred.append(train[i - s][0])
    return pred

def ARmod(train, test, p):
    Y = train[p:]
    X = []
    pred = []
    init_y = train[:p].reshape(-1)
    init_y = np.flip(init_y, 0)
    X.append(np.insert(init_y, 0, 1))
    for i in range(Y.shape[0] - 1):
        init_y = np.roll(init_y, 1)
        init_y[0] = Y[i][0]
        X.append(np.insert(init_y, 0, 1))
    beta = MultipleLinearRegression(np.asarray(X), Y)
    for i in range(test.shape[0]):
        init_y = np.roll(init_y, 1)
        init_y[0] = Y[-1][0]
        new_x = np.insert(init_y, 0, 1)
        pred.append(beta.T.dot(new_x)[0])
        X.append(new_x)
        Y = np.append(Y, test[i])
        Y = Y.reshape(-1, 1)
        beta = MultipleLinearRegression(np.asarray(X), Y)
    return pred


def SSE(y, y_hat):
    temp = 0
    for i in range(len(y)):
        temp += np.square(y[i]-y_hat[i])
    return(temp)

def MAPE(y, y_hat):
    temp = 0
    cnt = 0
    for i in range(len(y)):
        if y[i] != 0:
            temp += abs((y[i]-y_hat[i])/y[i])
            cnt++
    temp = (100/cnt)*temp
    return(temp)

def MultipleLinearRegression(X,Y):
    temp1 = X.T.dot(X)
    temp2 = X.T.dot(Y)
    temp1 = np.linalg.inv(temp1)
    beta_hat = temp1.dot(temp2)
    return(beta_hat)

# CDF
def plot_a(x):

    '''
    This function takes in samples and return the sample points
    and corresponding cdf
    '''

    n = len(x)
    x = sorted(x)
    x_a = []
    y_a = []
    y_curr = 0

    # Prepending 0's to make graph start from 0
    x_a.append(0)
    y_a.append(0)

    for i in x:
        y_curr += 1.0/n
        y_a.append(y_curr)
        x_a.append(i)
    # Removing prepended 0's from x_a and y_a
    x_a = x_a[1:]
    y_a = y_a[1:]

    return (x_a, y_a)
