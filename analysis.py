import numpy as np
from scipy import stats
import pandas as pd
import utils as ut
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Permutation Test
def PermutationTest(X, Y):
    tObs = abs(np.mean(X) - np.mean(Y))

    samples = np.append(X,Y)
    count = 0.0
    nFact = 100000 #(2 << (len(X) + len(Y)))
    for _ in range(nFact):
        tmp = np.random.permutation(samples)
        xP = tmp[:len(X)]
        yP = tmp[len(X):]
        tP = abs(np.mean(xP) - np.mean(yP))
        if tP > tObs:
            count += 1
    p = count / nFact
    return p

# Example use case: data, ['Year', 'Month]
def averageData(data, groupByFields):
    # Take Monthly Average
    normalizedData = data.groupby(groupByFields).mean()
    normalizedData = normalizedData.reset_index()
    return normalizedData

# Normalize data using Min Max scaling
def dataNormalization(nData):
    x = nData.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaledData = pd.DataFrame(x_scaled, columns = list(nData.columns))
    return scaledData

def analyzeCDFs(data, fields):
    for fld in fields:
        x, y = ut.plot_a(np.array(data[fld]))
        plt.plot(x, y)
        plt.title(fld)
        plt.show()

def plotD(fdata, gap, field):
    for year in [2011, 2012, 2013, 2014, 2015]:
        data = fdata[fdata['Year'] == year]
        vals = []
        i = 1
        while i <= 12:
            end = i + gap
            if end > 12:
                end = 12
            D1 = data[data['Month'] >= i]
            D2 = D1[D1['Month'] <= end][field].sum()
            #print(D2)
            i = i + gap
            vals.append(D2)
        plt.plot(vals)
    plt.title(field+ ' Gap :'+str(gap))
    plt.show()

    # plt.plot(D2['EGM'])
