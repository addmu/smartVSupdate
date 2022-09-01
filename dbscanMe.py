import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
#from LSSVM.lssvm import LSSVC
from sklearn import preprocessing
from scipy.signal import savgol_filter
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
import numpy.linalg as la
#from pykalman import KalmanFilter
#import simdkalman
from sklearn.neighbors import LocalOutlierFactor
import scipy.stats
from collections import deque
import itertools
from numpy import std, subtract, polyfit, sqrt, log
import math
from sklearn.covariance import ShrunkCovariance
from scipy.spatial import distance
import matplotlib
from sklearn.cluster import DBSCAN
from sklearn.utils.random import sample_without_replacement
from memory_profiler import memory_usage
from pympler import asizeof
import sys
import time
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.cluster import SpectralClustering,SpectralCoclustering,AgglomerativeClustering,SpectralBiclustering
#from mlxtend.plotting import heatmap
from sklearn import metrics
import hdbscan
from sklearn.preprocessing import MinMaxScaler
from sklearn import mixture

def readCSV(file):
    data = pd.read_csv(file,sep=',',header=None)
    data = data.to_numpy()
    #print(data[:,1])
    pos = []
    signal=[]
    for i in range(len(data[:,1])):
        pd0 = "{:08b}".format(int(data[:,1][i]))
        step_afe = int(pd0[-4:],2)
        pd1 = "{:08b}".format(int(data[:,2][i]))
        pd2 = "{:08b}".format(int(data[:,3][i]))
        sum_hex = int((pd1+pd2[0:4]),2)

        pd3="{:08b}".format(int(data[:,4][i]))
        pos.append(int((pd2[-3:]+pd3),2) * (1 if pd2[4]=='0' else -1))

        sum=(((sum_hex/2048)*1.8)*1000)/np.sqrt(2)**(step_afe)
        signal.append(format(sum,'.4f'))

    return pos,signal

def readSingle(file):
    pos, signal = readCSV(file)

    return pos, signal

def readAll(filelist):
    posAll = []
    signalAll = []
    for i in filelist:
        pos, signal = readCSV(i)
        posAll.extend(pos)
        signalAll.extend(signal)
    #print(posAll)
    return posAll,signalAll

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_getRealLabel(y, file,filelist):
    #def get_getRealLabel(y,file,filelist):
    #y = np.array(y)
    #x = np.arange(len(y))
    #numbers = np.arange(len(y))
    #plt.figure(figsize=(80,80))
    #plt.plot(x, y, color='blue')
    #for i in range(len(x)):
    #    plt.text(x[i], y[i], numbers[i])
    
    gt = [-1] * len(y)
    
    if file == filelist[0]:
        for i, n in enumerate(gt):
            if 75 < i < 120 or 220 < i < 293 or 400 < i < 500 or 675 < i < 722: #265
                gt[i] = 1
    elif file == filelist[1]:
        for i, n in enumerate(gt):
            if 120 < i < 160 or 303 < i < 355 or 525 < i < 548 or 734 < i < 762:#143
                gt[i] = 1
    elif file == filelist[2]:
        for i, n in enumerate(gt):
            if 120 < i < 140 or 220 < i < 294 or 390 < i < 414 or 545 < i < 578:#20+74+24+33
                gt[i] = 1
    elif file == filelist[3]:
        for i, n in enumerate(gt):
            if 63 < i < 82 or 203 < i < 250 or 360 < i < 420 or 585 < i < 634:#19+47+60+49
                gt[i] = 1
    elif file == filelist[4]:
        for i, n in enumerate(gt):
            if 100 < i < 131 or 270 < i < 540 or 670 < i < 725: #31+70+55
                gt[i] = 1
    elif file == filelist[5]:
        for i, n in enumerate(gt):
            if 45 < i < 100 or 255 < i < 378 or 532 < i < 635 or 814 < i < 867: #55+83+97+53
                gt[i] = 1
    elif file == filelist[6]:
        for i, n in enumerate(gt):
            if 130 < i < 869: #739
                gt[i] = 1
    elif file == filelist[7]:
        for i, n in enumerate(gt): #642
            if 120 < i < 762:
                gt[i] = 1

     
    #print(gt)
    return gt

def getAll_getRealLabel(filelist, group_size):
    gtAll = []
    count = 0
    for i in filelist:
        pos, signal = readCSV(i)
        if count ==0:
            pos = pos[group_size:]
        gt = get_getRealLabel(pos, i, filelist)
        gtAll.extend(gt)
        count += 1
        print(len(pos), i)
        gt = np.array(gt)
        mask = np.unique(gt)
        tmp = { }
        for v in mask:
            tmp[v] = np.sum(gt == v)
        print(mask)
        print(tmp)
    return gtAll

def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""

    # create the range of lag values
    i = len(ts) // 2
    lags = range(2, i)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst Exponent from the polyfit output
    return poly[0] * 2.0

def dataProcessing(pos, signal,group_size = 10):
    

    #import pdb;pdb.set_trace()
    '''
    pos = NormalizeData(pos[group_size:])
    signal = np.array(signal).astype(np.float64)
    signal = NormalizeData(signal[group_size:])
    X_train1 = np.squeeze(np.dstack((pos, signal)))
    
    X_train1 = np.squeeze(np.dstack((pos[group_size:], signal[group_size:])))
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train1)
    X_train1=scaler.transform(X_train1)
    '''
    X_train1 = np.squeeze(np.dstack((pos[group_size:], signal[group_size:])))
    #min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = preprocessing.MaxAbsScaler()
    #min_max_scaler = preprocessing.RobustScaler()
    #min_max_scaler = preprocessing.KernelCenterer()
    X_train1 = min_max_scaler.fit_transform(X_train1)

    return X_train1   
def coVariance(X): 
    ro, cl = X.shape
    row_mean = np.mean(X,axis=0)
    X_Mean = np.zeros_like(X)
    X_Mean[:] = row_mean     
    X_Minus = X - X_Mean
    covarMatrix = np.zeros((cl,cl))
    for i in range(cl):
        for j in range(cl):
            covarMatrix[i,j] = (X_Minus[:,i].dot(X_Minus[:,j].T)) / (ro-1)
    return covarMatrix

def hellinger_fast(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Fastest version.
    """
    #return sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p,q) ])

    n = len(p)
    sum = 0.0
    for i in range(n):
        sum += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    return result

def hellinger_fast_new(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Fastest version.
    """
    import pdb;pdb.set_trace()
    p_mean = np.mean(p, axis=0)
    q_mean = np.mean(q, axis=0)
    p_cov = ShrunkCovariance().fit(p)
    p_cov = p_cov.covariance_
    q_cov = ShrunkCovariance().fit(q)
    q_cov = q_cov.covariance_
    det_p = (np.linalg.det(p_cov))**0.25
    det_q = (np.linalg.det(q_cov))**0.25
    det_pq = np.linalg.det((p_cov+q_cov)/2)**0.5
    e = np.exp(np.dot(np.dot(((p_mean - q_mean).T), np.linalg.inv((p_cov+q_cov)/2)), (p_mean - q_mean)))
    #e = np.exp((-0.125)*((p_mean - q_mean).T)/((p_cov+q_cov)/2)*(p_mean - q_mean))
    
    BC = np.dot(det_p, det_q)/det_pq * e
    #result = np.sqrt(1 - BC)
    

    return 1-BC

def JensenShannonDivergence(p, q):
    #import pdb;pdb.set_trace()
    p = np.array(p)
    q = np.array(q)
    M = (p + q)/2
    return 0.5 * np.sum(p*np.log(p/M)) + 0.5 * np.sum(q*np.log(q/M))

def jsdiv(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    def _kldiv(A, B):
        return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)

    M = 0.5 * (P + Q)

    return 0.5 * (_kldiv(P, M) +_kldiv(Q, M))

def pdf(X, mean, sigma):
    numerator = np.exp(-0.5* np.dot(np.dot((X-mean),np.linalg.inv(sigma)),(X-mean).T))
    #import pdb;pdb.set_trace()
    denominator = math.sqrt(math.pow(2*np.pi,2)*np.linalg.det(sigma))
    #print(np.linalg.det(sigma))
    return numerator/denominator

def kalman_filter(sample1, sample2):
    #import pdb;pdb.set_trace()
    cov1 = np.zeros((2, 2))
    cov2 = np.zeros((2, 2))
    actual1, actual2 = sample1[0:2], sample2[0:2]
    mean1, mean2 = sample1[2:4], sample2[2:4]
    mean1, mean2 = np.reshape(mean1, (2,1)), np.reshape(mean2, (2,1))
    cov1[0,0],cov1[0,1], cov1[1,0], cov1[1,1] = sample1[4], sample1[5], sample1[6], sample1[7]
    cov2[0,0],cov2[0,1], cov2[1,0], cov2[1,1] = sample2[4], sample2[5], sample2[6], sample2[7]
    
    

    k = np.matmul(cov1, np.linalg.inv((cov1 + cov2))) #2*2
    mean = mean1 + np.dot(k, (mean2 - mean1))#2*1
    cov = cov1 - np.matmul(k, cov1)#2*2  #cov==0
    #import pdb;pdb.set_trace()
    mean = np.reshape(mean, (1,2))
    mean1, mean2 = np.reshape(mean1, (1,2)), np.reshape(mean2, (1,2))
    #p = pdf(actual1,mean,cov)*pdf(actual2,mean,cov)/(pdf(mean,mean,cov))**2 
    #p = pdf(actual1,mean,cov)*pdf(actual2,mean,cov)/(pdf(actual2,mean1,cov1)*pdf(actual1,mean2,cov2))
    #p = pdf(actual1,mean,cov)*pdf(actual2,mean,cov)/(pdf(actual1,mean1,cov1)*pdf(actual2,mean2,cov2))
    #p = 2*pdf(actual1,mean1,cov1)*pdf(actual2,mean2,cov2)/(pdf(actual1,mean1,cov1)**2 + pdf(actual2,mean2,cov2)**2)
    #p = 2*pdf(actual1,mean,cov)*pdf(actual2,mean,cov)/(pdf(actual1,mean1,cov1)**2 + pdf(actual2,mean2,cov2)**2)
    p = hellinger_fast(actual1, actual2)
    #p = distance.jensenshannon(actual1, actual2)
    #p = JensenShannonDivergence(actual1, actual2)
    print(p)
    #if np.isnan(p[0,0]):
    #    print("true")
    #    pro = 0
    #else:
    #    pro = p[0,0]
    #pro = p[0,0]

    return p

def JensenShannon_kernel(z_measure, y=None):
    z_measure1 = z_measure[:, 0] #pos
    z_measure2 = z_measure[:, 1] #signal

    mean_pos = np.array([])
    mean_signal = np.array([])
    sigma_1_1 = np.array([])
    sigma_1_2 = np.array([])
    sigma_2_1 = np.array([])
    sigma_2_2 = np.array([])
    group_pos = np.array([])
    group_signal = np.array([])
    for i in tqdm(range(10, len(z_measure1))):
        if i < 10:
            continue
        #if i == 0:
            #g_pos = [z_measure1[0], z_measure1[0]]
            #g_signal = [z_measure2[0], z_measure2[0]]
        else:
            g_pos = z_measure1[max(0, i-10):i+1]
            g_signal = z_measure2[max(0, i-10):i+1]
        if i == 10:
            group_pos = g_pos
            group_signal = g_signal
        else:
            group_pos = np.vstack((group_pos, g_pos))
            group_signal = np.vstack((group_signal, g_signal))
        
        mean_pos = np.append(mean_pos,np.mean(g_pos))
        mean_signal = np.append(mean_signal,np.mean(g_signal))
        #import pdb;pdb.set_trace()
        y = np.squeeze(np.dstack((g_pos, g_signal)))
        #cov = coVariance(y)
        #y = np.reshape(y, (2,11))
        #cov = np.cov(y)
        covS = ShrunkCovariance().fit(y)
        cov = covS.covariance_
        sigma_1_1 = np.append(sigma_1_1,cov[0,0])
        sigma_1_2 = np.append(sigma_1_2,cov[0,1])
        sigma_2_1 = np.append(sigma_2_1,cov[1,0])
        sigma_2_2 = np.append(sigma_2_2,cov[1,1])
    group_pos_signal = np.squeeze(np.dstack((group_pos, group_signal)))
    #checkernel(group_pos_signal)
    k_corr = np.zeros((group_pos_signal.shape[0], group_pos_signal.shape[0]))
    for i in tqdm(range(group_pos_signal.shape[0])):
        for j in range(group_pos_signal.shape[0]):
            
            g_i = np.squeeze(np.reshape(group_pos_signal[i, :], (1, 22)))
            g_j = np.squeeze(np.reshape(group_pos_signal[j, :], (1, 22)))
            k_corr[i,j] = jsdiv(g_i, g_j)

    return k_corr

def hellinger_kernel(z_measure, y=None):
    z_measure1 = z_measure[:, 0] #pos
    z_measure2 = z_measure[:, 1] #signal

    mean_pos = np.array([])
    mean_signal = np.array([])
    sigma_1_1 = np.array([])
    sigma_1_2 = np.array([])
    sigma_2_1 = np.array([])
    sigma_2_2 = np.array([])
    group_pos = np.array([])
    group_signal = np.array([])
    for i in tqdm(range(10, len(z_measure1))):
        if i < 10:
            continue
        #if i == 0:
            #g_pos = [z_measure1[0], z_measure1[0]]
            #g_signal = [z_measure2[0], z_measure2[0]]
        else:
            g_pos = z_measure1[max(0, i-10):i+1]
            g_signal = z_measure2[max(0, i-10):i+1]
        if i == 10:
            group_pos = g_pos
            group_signal = g_signal
        else:
            group_pos = np.vstack((group_pos, g_pos))
            group_signal = np.vstack((group_signal, g_signal))
        
        mean_pos = np.append(mean_pos,np.mean(g_pos))
        mean_signal = np.append(mean_signal,np.mean(g_signal))
        #import pdb;pdb.set_trace()
        y = np.squeeze(np.dstack((g_pos, g_signal)))
        #cov = coVariance(y)
        #y = np.reshape(y, (2,11))
        #cov = np.cov(y)
        covS = ShrunkCovariance().fit(y)
        cov = covS.covariance_
        sigma_1_1 = np.append(sigma_1_1,cov[0,0])
        sigma_1_2 = np.append(sigma_1_2,cov[0,1])
        sigma_2_1 = np.append(sigma_2_1,cov[1,0])
        sigma_2_2 = np.append(sigma_2_2,cov[1,1])
    group_pos_signal = np.squeeze(np.dstack((group_pos, group_signal)))
    #checkernel(group_pos_signal)
    k_corr = np.zeros((group_pos_signal.shape[0], group_pos_signal.shape[0]))
    for i in tqdm(range(group_pos_signal.shape[0])):
        for j in range(group_pos_signal.shape[0]):
            
            g_i = np.squeeze(np.reshape(group_pos_signal[i, :], (1, 22)))
            g_j = np.squeeze(np.reshape(group_pos_signal[j, :], (1, 22)))
            k_corr[i,j] = hellinger_fast(g_i, g_j)

    return k_corr

def custom_hellinger_dist(p,q):
    #import pdb;pdb.set_trace()
    n = len(p)
    sum = 0.0
    for i in range(n):
        sum += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
    #group_r = np.append(group_r, result)
    #print(result)
    #count+=1
    return result

def checkernel(group_pos_signal):
    #import pdb;pdb.set_trace()
    k_corr = np.zeros((group_pos_signal.shape[0], group_pos_signal.shape[0]))
    for i in tqdm(range(group_pos_signal.shape[0])):
        for j in range(group_pos_signal.shape[0]):
            #g_i = group_pos_signal[i, :]
            #g_j = group_pos_signal[j, :]
            g_i = np.squeeze(np.reshape(group_pos_signal[i, :], (1, 22)))
            g_j = np.squeeze(np.reshape(group_pos_signal[j, :], (1, 22)))
            k_corr[i,j] = hellinger_fast(g_i, g_j)
            
            #k_corr[i,j] = hellinger_fast(group_pos_signal[i, :], group_pos_signal[j, :])
    #plt.figure(figsize = (20, 20))
    #ax = sns.heatmap(k_corr)
    #plt.show()
    #heatmap(k_corr, figsize=(60, 60))
    '''
    heatmap = plt.pcolor(k_corr)

    for y in tqdm(range(k_corr.shape[0])):
        for x in range(k_corr.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.4f' % k_corr[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

    plt.colorbar(heatmap)
    plt.show()
    '''
    #import pdb;pdb.set_trace()
    return k_corr
    
        

def kalman_kernel(z_measure, y=None): 
    #import pdb;pdb.set_trace()
    z_measure1 = z_measure[:, 0] #pos
    z_measure2 = z_measure[:, 1] #signal

    mean_pos = np.array([])
    mean_signal = np.array([])
    sigma_1_1 = np.array([])
    sigma_1_2 = np.array([])
    sigma_2_1 = np.array([])
    sigma_2_2 = np.array([])
    group_pos = np.array([])
    group_signal = np.array([])
    for i in tqdm(range(10, len(z_measure1))):
        if i < 10:
            continue
        #if i == 0:
            #g_pos = [z_measure1[0], z_measure1[0]]
            #g_signal = [z_measure2[0], z_measure2[0]]
        else:
            g_pos = z_measure1[max(0, i-10):i+1]
            g_signal = z_measure2[max(0, i-10):i+1]
        if i == 10:
            group_pos = g_pos
            group_signal = g_signal
        else:
            group_pos = np.vstack((group_pos, g_pos))
            group_signal = np.vstack((group_signal, g_signal))
        
        mean_pos = np.append(mean_pos,np.mean(g_pos))
        mean_signal = np.append(mean_signal,np.mean(g_signal))
        #import pdb;pdb.set_trace()
        y = np.squeeze(np.dstack((g_pos, g_signal)))
        #cov = coVariance(y)
        #y = np.reshape(y, (2,11))
        #cov = np.cov(y)
        covS = ShrunkCovariance().fit(y)
        cov = covS.covariance_
        sigma_1_1 = np.append(sigma_1_1,cov[0,0])
        sigma_1_2 = np.append(sigma_1_2,cov[0,1])
        sigma_2_1 = np.append(sigma_2_1,cov[1,0])
        sigma_2_2 = np.append(sigma_2_2,cov[1,1])
    group_pos_signal = np.squeeze(np.dstack((group_pos, group_signal)))
    #checkernel(group_pos_signal)
    
    training_set = np.squeeze(np.dstack((z_measure1[10:], z_measure2[10:], mean_pos, mean_signal, sigma_1_1, sigma_1_2, sigma_2_1, sigma_2_2)))
    #import pdb;pdb.set_trace()
    print(training_set.shape)

    k_value = np.zeros((training_set.shape[0], training_set.shape[0]))
    for i in tqdm(range(training_set.shape[0])): 
        for j in tqdm(range(training_set.shape[0])): #in total for 1000 rows
            #import pdb;pdb.set_trace()
            k_value[i,j] = kalman_filter(training_set[i, :], training_set[j, :]) # fill each row, and it's a symmetric matrix
    print('k value is:',k_value)
    print(k_value.shape)
    return k_value




        

def radial_basis(x, y, gamma=10):
    return np.exp(-gamma * la.norm(np.subtract(x, y)))

def proxy_kernel(X, Y, K=radial_basis):
    """Another function to return the gram_matrix,
    which is needed in SVC's kernel or fit
    """
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = K(x, y)
    return gram_matrix

#def gaussian_kernel(X, Y):
#    kernel = euclidean_distances(X, Y) ** 2
#    kernel = kernel*(-1/(self.gamma**2))
#    kernel = np.exp(kernel)
#    return kernel


def plot_ori_datadistribution(x,y):
    fig, ax = plt.subplots()
    ax.scatter(x,y, s=4, color="g")

def plot_scatter_three_line(y, labels):
    y = np.array(y)
    x = np.arange(len(y))
    plt.figure(figsize=(60,60))
    plt.plot(x, y, color='blue')
    threshold = labels < 0
    below_threshold = y < threshold
    print(labels)
    #import pdb;pdb.set_trace()
    #print(threshold)
    threshold1 = labels == 0
    threshold2 = labels == 1
    threshold3 = labels == 2
    plt.scatter(x[threshold1], y[threshold1], color='red') 
    plt.scatter(x[threshold2], y[threshold2], color='green') 
    plt.scatter(x[threshold3], y[threshold3], color='purple') 
    #plt.scatter(x[threshold], y[threshold], color='red') 
    #above_threshold = np.logical_not(threshold)
    #plt.scatter(x[above_threshold], y[above_threshold], color='green') 

def plot_scatter_line(y,label):
    y = np.array(y)
    x = np.arange(len(y))
    plt.figure(figsize=(60,60))
    plt.plot(x, y, color='blue')
    threshold = label < 0
    below_threshold = y < threshold
    #print(threshold)
    plt.scatter(x[threshold], y[threshold], color='red') 
    above_threshold = np.logical_not(threshold)
    plt.scatter(x[above_threshold], y[above_threshold], color='green') 

def plot_pred_label(X_train,label,u_labels):
    plt.figure()
    for i in u_labels:
        plt.scatter(X_train[label == i , 0] , X_train[label == i , 1] , label = i)
    plt.legend()

def plot_text(y):
    y = np.array(y)
    x = np.arange(len(y))
    numbers = np.arange(len(y))
    plt.figure(figsize=(80,80))
    plt.plot(x, y, color='blue')
    for i in range(len(x)):
        plt.text(x[i], y[i], numbers[i])
    plt.show()
def _ema(arr):
    N = len(arr)
    α = 2/(N+1)
    data = np.zeros(len(arr))
    for i in range(len(data)):
        data[i] = arr[i] if i==0 else α*arr[i]+(1-α)*data[i-1] 
    return data[-1]

def EMA(arr,period=10):
    data = np.full(arr.shape,np.nan)
    for i in range(period-1,len(arr)):
        data[i] = _ema(arr[i+1-period:i+1])
    return data
def moving_average(data_array, n):

    it = iter(data_array)
    d = deque(itertools.islice(it, n - 1))
    print(d)
    s = sum(d)
    d.appendleft(0)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n

def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.
    
    Denoises data using the fast fourier transform.
    
    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)
        
    Returns
    -------
    clean_data : numpy.array
        The denoised data.
        
    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton
    
    """
    n = len(x)
    
    # compute the fft
    fft = np.fft.fft(x, n)
    
    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n
    
    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft
    # inverse fourier transform
    clean_data = np.fft.ifft(fft)
    
    if to_real:
        clean_data = clean_data.real
    
    return clean_data
def matrixtoarray(k):
    #k = np.random.randint(0,10,(4,4))
    k_0 = np.array([])
    k_1 = np.array([])
    
    for i in tqdm(range(k.shape[0])):
        for j in range(i, k.shape[0]):
            
            k_0 = np.append(k_0, k[k.shape[0]-1, i])
            k_1 = np.append(k_1, k[j, i])
    X_train = np.squeeze(np.dstack((k_0, k_1)))
    #import pdb;pdb.set_trace()
    return X_train


if __name__ == '__main__':
    file = 'test_13062022/test7_empty_13062022.csv'
    filelist= ['test_13062022/test1_box_full_13062022.csv', 'test_13062022/test2_box_full_13062022.csv','test_13062022/test3_missing2_13062022.csv',
                'test_13062022/test4_missing1-2-1_13062022.csv', 'test_13062022/test5_missing2-4-1_13062022.csv','test_13062022/test6_missing2-4-1-obj_13062022.csv',
                'test_13062022/test7_empty_13062022.csv','test_13062022/test8_empty2_13062022.csv']
    pos_test, signal_test_ori = readSingle(file)
    #pos, signal_ori = readAll(filelist)
    gt_test = get_getRealLabel(pos_test,file,filelist)   # for getting real label
    #get_getRealLabel(pos,file,filelist)   # for getting real label
    group_size = 10
    #gt = getAll_getRealLabel(filelist,group_size)

    X_train = dataProcessing(pos_test, signal_test_ori,group_size)
    z_measure1 = X_train[:, 0]
    z_measure2 = X_train[:, 1]

    group_pos = np.array([])
    group_signal = np.array([])
    for i in tqdm(range(10, len(z_measure1))):
        if i < 10:
            continue
        #if i == 0:
            #g_pos = [z_measure1[0], z_measure1[0]]
            #g_signal = [z_measure2[0], z_measure2[0]]
        else:
            g_pos = z_measure1[max(0, i-10):i+1]
            g_signal = z_measure2[max(0, i-10):i+1]
            
        if i == 10:
            group_pos = g_pos
            group_signal = g_signal
        else:
            group_pos = np.vstack((group_pos, g_pos))
            group_signal = np.vstack((group_signal, g_signal))
        
    #import pdb;pdb.set_trace()
    group_pos_signal = np.squeeze(np.dstack((group_pos, group_signal)))
    group_ps = np.reshape(group_pos_signal, (group_pos_signal.shape[0], group_pos_signal.shape[1]*group_pos_signal.shape[2]))
    #import pdb;pdb.set_trace()
    start_a = time.time()
    #k_value = checkernel(group_pos_signal)
    #group_ps = k_value

    
    #X_train = k_value #[980,980]
    #X_train = matrixtoarray(k_value)
    #sample_without_replacement()

    #plt.scatter(X_train[:, 0], X_train[:, 1])
    #plt.show()
    #import pdb;pdb.set_trace()
    #k_value = k_value[:5,:5]
    '''
    k_value = X_train
    
    intervals = 10
    threshold = 0
    clique_instance = clique(k_value, intervals, threshold)

    clique_instance.process()
    clique_cluster = clique_instance.get_clusters()  # allocated clusters

    noise = clique_instance.get_noise()
    cells = clique_instance.get_cells() 

    print("Amount of clusters:", len(clique_cluster))
    print(clique_cluster)

    clique_visualizer.show_grid(cells, k_value) 
    clique_visualizer.show_clusters(k_value, clique_cluster, noise)  # show clustering resultsf
    '''
    #pca = PCA(n_components=2).fit(X_train)
    #pca_5d = pca.transform(X_train)
    #X_train = pca_5d
    '''
    res = []
    for eps in tqdm(np.arange(0.001, 6, 0.05)):
        for min_samples in range(2, 9):
            #import pdb;pdb.set_trace()
            #db = DBSCAN(eps = eps, min_samples = min_samples, metric = custom_hellinger_dist).fit(group_ps)
            db = DBSCAN(eps = eps, min_samples = min_samples).fit(group_ps)
            n_clusters = len([i for i in set(db.labels_) if i != -1])
            outliners = np.sum(np.where(db.labels_ == -1,1,0))
            stats = str(pd.Series([i for i in db.labels_ if i != -1]).value_counts().values)
            res.append({'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'outliners': outliners, 'stats': stats})
    df = pd.DataFrame(res)
    pd.set_option('display.max_rows', None)
    print(df.loc[df.n_clusters == 2, :])

    '''

    #group_ps = np.array(group_ps).T
    #group_ps = checkernel(group_pos_signal)
    start = time.time()
    #print(group_ps)
    #db = DBSCAN(eps=3.601, min_samples=4).fit(group_ps)#5(0.14,3)
    #db = DBSCAN(eps=0.2, min_samples=9, metric = custom_hellinger_dist).fit(group_ps)#5(0.14,3)
    #db = OPTICS(min_samples=25).fit(group_ps)
    #db = hdbscan.HDBSCAN(min_cluster_size=10)
    #db = SpectralClustering(n_clusters=2, assign_labels='discretize', affinity='rbf', random_state=3).fit(group_ps)
    #db = SpectralClustering(n_clusters=3,assign_labels='discretize',affinity='nearest_neighbors',n_neighbors=36,random_state=3).fit(group_ps)
    #db = SpectralClustering(n_clusters=3, assign_labels='discretize', affinity='precomputed_nearest_neighbors', random_state=3).fit(group_ps)
    db = SpectralCoclustering(n_clusters=3, svd_method='arpack', random_state=0).fit(group_ps)
    #db = AgglomerativeClustering(n_clusters=3,linkage='complete').fit(group_ps)
    #db = mixture.BayesianGaussianMixture(n_components=2, covariance_type='full', weight_concentration_prior_type='dirichlet_process',
    #init_params="kmeans", weight_concentration_prior=0.7, random_state=2).fit(group_ps)
    #db = mixture.BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.01, init_params='kmeans', random_state=2).fit(group_ps)
    #bay_gmm_weights = db.weights_
    #print(np.round(bay_gmm_weights, 2))
    #n_clusters_ = (np.round(bay_gmm_weights, 2) > 0).sum()
    #print('Estimated number of clusters: ' + str(n_clusters_))
    #import pdb;pdb.set_trace()
    #db = SpectralBiclustering(n_clusters=2, method="bistochastic", random_state=0)

    end = time.time()
    print('model time: ', end - start)
    mem_fit = memory_usage((db.fit, (group_ps,)),max_usage = True)
    print('memory caused for train: ', mem_fit)
    print('the size of the model: ', asizeof.asizeof(db))
    #print('memory caused for train: ', mem_fit)
    #print('all time: ', end - start_a)
    #db = OPTICS(min_samples=5, metric = custom_hellinger_dist).fit(group_ps)
    #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    #core_samples_mask[db.core_sample_indices_] = True
    #labels = db.labels_
    labels = db.row_labels_ 
    #labels = db.predict(group_ps)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    #print(labels)
    #import pdb;pdb.set_trace()
    gt_test = np.array(gt_test)
    
    gt_test[gt_test == 1] = 0
    gt_test[gt_test == -1] = 1
    print('acc: ', metrics.rand_score(gt_test[group_size*2:], labels))
    plot_scatter_three_line(pos_test[group_size*2:], labels) 
    
    
    
    '''
    for i in range(0, group_ps.shape[0]):
        if db.row_labels_[i] == 0:
            c1 = plt.scatter(group_ps[i, 0], group_ps[i, 1], c='r', marker='+')
        elif db.row_labels_[i] == 1:c2 = plt.scatter(group_ps[i, 0], group_ps[i, 1], c='g', marker='o')
        elif db.row_labels_[i] == 2:c3 = plt.scatter(group_ps[i, 0], group_ps[i, 1], c='b', marker='o')


    plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Cluster 3'])
    plt.title('DBSCAN finds 3 clusters and Noise')
    '''

    '''
    pca = PCA(n_components=2).fit(group_ps)
    pca_2d = pca.transform(group_ps)
    # Plot based on Class
    for i in range(0, pca_2d.sythohape[0]):
        if db.labels_[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif db.labels_[i] == 1:c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif db.labels_[i] == 2:c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='o')
        elif db.labels_[i] == -1:
            c4 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='k', marker='*')


    plt.legend([c1, c2, c3, c4], ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Noise'])
    plt.title('DBSCAN finds 3 clusters and Noise')
    '''
    
    
    

    '''
    X_train = X_train[:, :2]
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    #print(labels)
    #print(labels.shape)
    gt_test = gt_test[group_size:]
    fig, ax = plt.subplots(figsize = (80,80))
    ax.scatter(X_train[:, 0], X_train[:, 1], s = 4, color = "g")
    ax.scatter(X_train[labels == -1, 0], X_train[labels == -1, 1],  
            facecolors="none", edgecolors = "red", s = 80, label = "predicted outliers")
    ax.scatter(X_train[gt_test == -1, 0], X_train[gt_test == -1, 1], marker = "x", 
            color="r", s=40, label="ground truth outliers")

    plot_scatter_three_line(pos_test[group_size*2:], labels) 
    '''
    

    '''
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
        # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X_train[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X_train[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    
    '''

    #plot_text(pos_test)
    #import pdb;pdb.set_trace()
    
    '''
    signal_test_ori = map(float, signal_test_ori)
    signal_ori = map(float, signal_ori)
    signal_test = [i * 1 for i in signal_test_ori]
    signal = [i * 1 for i in signal_ori]
    '''
    #use fourier filter
    '''
    pos_test = np.array(pos_test)
    signal_test_ori = np.array(signal_test_ori)
    X = scipy.fft.fft(signal_test_ori.reshape((-1,)))
    freqs = scipy.fft.fftfreq(len(signal_test_ori),1)
    fig=plt.figure(figsize=(20,12))
    plt.plot(freqs,np.abs(X))
    
    x = np.arange(len(pos_test))
    plt.figure(figsize=(60,60))
    plt.plot(x, pos_test, color='blue')

    pos_test = fft_denoiser(pos_test, len(pos_test)*1.3)
    signal_test_ori = fft_denoiser(signal_test_ori, len(signal_test_ori)*1.3) # *1.5  1.8

    x = np.arange(len(pos_test))
    plt.figure(figsize=(60,60))
    plt.plot(x, pos_test, color='blue')
    


    '''

            
    #import pdb; pdb.set_trace()
    #print(X_train)
    #print(len(groups_norm_pos[:10]))
    #yhat = savgol_filter(groups_norm_pos[:10], 9, 3) 
    #print(yhat)
    #x = np.linspace(0,10,10)
    
    #plt.plot(x,groups_norm_pos[:10])
    #plt.plot(x,yhat, color='red')
    #plt.show()

    #X_train = dataProcessing(pos, signal,group_size)
    #X_test = dataProcessing(pos_test, signal_test,group_size)
    #X_train = dataProcessing(pos_test, signal_test_ori,group_size) 
    #X_train = dataProcessing(pos, signal_ori, group_size)

    #X_train = np.squeeze(np.dstack((pos_test, signal_test_ori)))
    #scaler = StandardScaler(with_mean=True, with_std=True)
    #scaler.fit(X_train)
    #X_train=scaler.transform(X_train)

    #clf = svm.OneClassSVM(kernel='rbf',gamma='scale',nu=0.2)  
    #clf = svm.OneClassSVM(kernel='precomputed')
    #print(clf.kernel)
    #clf = EllipticEnvelope(assume_centered=True, contamination =0.3)
    #print(" out: ",X_train.shape[0], X_train.shape[1])
    #clf.fit(hellinger_kernel(X_train))
    #label = clf.fit_predict(hellinger_kernel(X_train))
    
    #label = clf.predict(hellinger_kernel(X_train))
    #u_labels = np.unique(label)  

    #print(len(X_train), len(label))

    
    #plot_scatter_line(pos_test, label)
    #plot_scatter_line(pos_test[group_size*2:], label) 
    #plot_scatter_line(pos_test, label) 
    #plot_pred_label(X_train,label,u_labels)
    '''
    X_train = X_train[group_size:]
    fig, ax = plt.subplots(figsize = (80,80))
    ax.scatter(X_train[:, 0], X_train[:, 1], s = 4, color = "g")
    ax.scatter(X_train[label == -1, 0], X_train[label == -1, 1],  
            facecolors="none", edgecolors = "red", s = 80, label = "predicted outliers")
    ax.scatter(X_train[gt_test == -1, 0], X_train[gt_test == -1, 1], marker = "x", 
            color="r", s=40, label="ground truth outliers")
    '''
    
    
    '''
    X_train = np.squeeze(np.dstack((pos,signal)))
    #print(X_train)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    #clf = KMeans(n_clusters=2)
    #clf = LocalOutlierFactor(n_neighbors=10,contamination=0.5)
    #clf = EllipticEnvelope(assume_centered=True, contamination =0.5)
    clf = svm.OneClassSVM(kernel= 'rbf',gamma='auto',nu=0.075)
    clf = LSSVC(gamma=1, kernel='linear')
    label = clf.fit_predict(X_train)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(X_train[label == i , 0] , X_train[label == i , 1] , label = i)
    plt.legend()
    plt.show()
    '''
    '''
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 0], X_train[:, 1], s=4, color="g")

    ax.set_xlabel('Distance')
    ax.set_ylabel('Energy')
    ax.set_title('All data')
    ax.legend(loc="lower right")
    '''


    plt.show()
