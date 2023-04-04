### Libraries
import numpy as np
import pandas as pd
import copy
from scipy.stats import mode
import seaborn as sns
import warnings
import sys
#from numba import njit, prange
from os import listdir
from tqdm import tqdm

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import art3d

### Utility functions
#Getting Class
def get_class(data,centroids,K):
    size = data.shape[0]
    data_class = np.zeros([size]) #Create an array to store classes of each data
    dist_mat = np.zeros((size,K))
    for k in range(K):
      dist_mat[:,k] = np.sqrt(np.sum((data-centroids[k,:])**2,axis = 1))
    #Wherever the minimum
    data_class = np.argmin(dist_mat,1).reshape(-1,1)
    return data_class.ravel()

# Getting centroids
def get_centroids(data,data_class,K):
    dim = data.shape[1] #The number of columns in data
    centroids = np.zeros((K,dim))
    
    for i in range(K):
        class_data = data[data_class==i]
        centroids[i,:] = np.array(np.mean(class_data,0)).reshape(1,-1)

    return centroids

#K-means
def kmeans(data, num_clus, tol  = 10e-6):
    """
    Performs kmeans clustering and returns parameter initialization for GMM
    """
    
    size,num_feat = data.shape[0],data.shape[1]
    np.random.seed(5546)
    idx = np.random.choice(np.array(range(size)), size=num_clus,replace = False)
    try:
      cent_ = data.loc[idx] 
    except:
      cent_ = data[idx,:]
    centroids = copy.deepcopy(cent_)

    ite = 0 
    
    while np.sum(np.linalg.norm(abs(cent_-centroids),2,axis=1)<tol)!=num_clus or ite == 0:
        ite += 1
        cent_ = copy.deepcopy(centroids)
        data_class = get_class(data,centroids,num_clus)
        centroids = get_centroids(data,data_class,num_clus)
    centroids = centroids.T
    kmeans_result = {}
    kmeans_result['centroids'] = centroids
    kmeans_result['num_points'] = {}
    kmeans_result['cov'] = {}
    for q in range(num_clus):
      kmeans_result['num_points'][q] = len(data_class[data_class==q])
      tmp = (data[data_class==q,:].T-centroids[:,q].reshape(-1,1))
      kmeans_result['cov'][q] = (1/kmeans_result['num_points'][q])*np.matmul(tmp,tmp.T)
    return kmeans_result  

### Gaussian Mixture Model - Bayes Classifier
def initialize(X, Q, case):
  """
  Performs kmeans for each class for GMM initialization
  X - Training Data (preferably numpy array)
  y - Training Labels
  classes - list of classes in order
  Q - Number of clusters for each class

  returns - initial set of parameters
  """
  init_params = {}
  
  classes = np.array([0])
  y = np.array([0 for i in range(len(X))])

  for c in classes:
    X_c = X[(y==c).ravel(),:]
    y_c = y[y==c]
    init_params[c] = {}
    kmeans_result = kmeans(X,Q)
    for q in range(Q):
      init_params[c][q] = {}
      init_params[c][q]['w'] = kmeans_result['num_points'][q]/len(y_c)
      init_params[c][q]['mu'] = kmeans_result['centroids'][:,q].reshape(-1,1)
      cov = kmeans_result['cov'][q]
      if case == "diag":
        cov = np.diag(np.diag(cov))
      init_params[c][q]['cov'] = cov
  return init_params[0]

def gmm(X,Q,case = "full",tol = 1e-8):
  """
  Trains/Fits GMM Model for each class using Expectation Maximization Algorithm
  X - Training Data (preferably numpy array)
  y - Training Labels
  classes - list of classes in order
  Q - Number of clusters for each class

  returns - final parameter set : params, predicted data classes, Posterior
  """
  classes = np.array([0])
  y = np.array([0 for i in range(len(X))])

  n,d = np.shape(X)

  #Initializing parameters for each class using kmeans
  init_params = initialize(X,Q,case)

  #Stores parameters (w,mu,cov) for each class c and each subclass q
  params = {}

  #Class Prior Probabilities
  class_support = np.array([len(y[y==c])/n for c in classes]).reshape(1,-1)
  labels = {i:c for i,c in zip(range(len(classes)),classes)}

  for i in labels:
    c = labels[i]
    ite = 0

    params = {}

    log_prob_a = 0  #log likelihood before update
    log_prob_b = 0  #log likelihood after update

    X_c = X[(y==c).ravel(),:] #Data belonging to class c
    n_c = X_c.shape[0]  

    while (abs(log_prob_a-log_prob_b)>tol or ite == 0) and ite<100:

      #***Expectation Step - Calculating Posterior***

      posterior = np.zeros((n_c,Q))
      log_prob_b = log_prob_a
      log_prob_a = 0

      #For each Component
      for q in range(Q):
        if ite == 0:
          mu_q = init_params[q]['mu']
          cov_q = init_params[q]['cov']
          w_q = init_params[q]['w']
        else:
          mu_q = params[q]['mu']
          cov_q = params[q]['cov']
          w_q = params[q]['w']

        min_eig = 1
        det =  np.linalg.det(cov_q)

        #Inspecting the Covariance Matrix and enforcing non-singularity and positive definitiveness
        while abs(det)<1e-12 or det<0 or min_eig < 0:
          cov_q += abs(np.diag((np.random.normal(0,2,(d,1))).ravel()))  ##Addition of Gaussian Noise
          det = np.linalg.det(cov_q)
          if det > 0:
            min_eig = np.min(np.real(np.linalg.eigvals(cov_q)))         ##Minimum Eigenvalue
        
        icov_q = np.linalg.inv(cov_q)  ##Inverse of Covariance Matrix

        #Calculating likelihood given component/cluster
        #X -> (nxd), mu_q -> (dx1), cov_q -> (dxd), finally, tmp -> (nx1)
        tmp  = np.sum(np.dot((X_c.T-mu_q).T,icov_q) * (X_c.T-mu_q).T,axis=1)
        # pX_q_c (log-likelihood) -> (nx1)
        pX_q_c = (1/((2*np.pi)**(d/2))) * (np.linalg.det(cov_q)**-0.5) * np.exp(-0.5*tmp) 
        
        #Enforcing Bounds on the likelihood values to prevent small, large or nan values
        pX_q_c[pX_q_c<1e-8] = 1e-8
        pX_q_c[pX_q_c==np.inf] = 1
        pX_q_c[np.isnan(pX_q_c)] = 1e-8

        posterior[:,q] = w_q*pX_q_c
        log_prob_a += w_q*pX_q_c ##(nx1 + ... q Q times) 

      #Calculating Total Log Probability
      log_prob_a = np.sum(np.log(log_prob_a))  ##(nx1)

      #Calculating Posterior gamma -> (nxQ)
      posterior /= np.sum(posterior,axis=1).reshape(-1,1)
      
      #***Maximization Step - Updating Parameters***
      #w_all -> (1xQ)
      w_all = np.sum(posterior,axis = 0)/n_c
      # print(w_all)
      #mu_all -> (dxQ)
      mu_all = (1/(w_all*n_c))*(np.matmul(posterior.T,X_c)).T
      
      for q in range(Q):
        params[q] = {}
        params[q]['w'] = w_all[q]
        params[q]['mu'] = mu_all[:,q].reshape(-1,1)
        #tmp -> (dxd)
        tmp  = np.matmul(np.matmul(X_c.T-params[q]['mu'],np.diag(posterior[:,q])),(X_c.T-params[q]['mu']).T)
        if case == "diag":
          tmp = np.diag(np.diag(tmp))

        tmp2  = w_all[q]*n_c

        params[q]['cov'] = tmp/(tmp2)
      ite += 1 
  params['class_support'] = class_support
  params['Q'] = Q

  #Predicting Training Labels and Class-Conditional Posterior (without dividing by Evidence) with Training Set
  #pred_classes,class_post = gmm_classifier_predict(X,classes,params)
  return params

def gmm_classifier_predict(X,params):
  """
  Function that predicts labels using trained model parameters.
  X - Data
  classes - list of classes in order
  params - trained model parameters
  Returns - predicted labels, class-wise posterior (without normalizing)
  """
  classes = np.array([0])
  n,d = X.shape
  class_support = params['class_support']
  labels = {i:c for i,c in zip(range(len(classes)),classes)}
  likelihoods = np.zeros((n,len(classes)))
  Q = params['Q']

  for i in labels:
    c = labels[i] 
    for q in range(Q):
      mu_q = params[q]['mu']
      cov_q = params[q]['cov']
      w_q = params[q]['w']
      
      icov_q = np.linalg.inv(cov_q)

      tmp  = np.sum(np.dot((X.T-mu_q).T,icov_q) * (X.T-mu_q).T,axis=1)
      pX_q_c = (1/((2*np.pi)**(d/2))) * (np.linalg.det(cov_q)**-0.5) * np.exp(-0.5*tmp)
      likelihoods[:,i] += pX_q_c * w_q
  
  class_post = np.log(likelihoods*class_support)
  pred_labels = np.argmax(class_post,axis=1)
  pred_classes = np.array([labels[i] for i in pred_labels]).reshape(-1,1)

  return pred_classes, class_post 


def distribution(params, X, Y):
    """
    params: {component:{mu: mean, cov: covariance, w: weight}}
    """
    L = []
    for i in range(len(X)):
        for j in range(len(X[i])):
            L.append([X[i][j], Y[i][j]])
    L = np.array(L)
    Q = params['Q']
    for i in range(Q):
        mu_q = params[i]['mu']
        cov_q = params[i]['cov']
        w_q = params[i]['w']
        d = len(mu_q)
        icov_q = np.linalg.inv(cov_q)
        #X -> (nxd), mu_q -> (dx1), cov_q -> (dxd), finally, tmp -> (nx1)
        tmp  = np.sum(np.dot((L.T-mu_q).T,icov_q) * (L.T-mu_q).T,axis=1)
        # pX_q_c (log-likelihood) -> (nx1)o
        if i ==0:
            pX_q_c = w_q * (1/((2*np.pi)**(d/2))) * (np.linalg.det(cov_q)**-0.5) * np.exp(-0.5*tmp)
        else:
            pX_q_c += w_q * (1/((2*np.pi)**(d/2))) * (np.linalg.det(cov_q)**-0.5) * np.exp(-0.5*tmp)
    return pX_q_c.reshape((len(X), len(Y)))

def draw_ellipse(position, covariance, edge_color, ax=None):
    """
    position: mean
    covariance
    """
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, fc = 'none', ec = edge_color))

def draw_ellipse_3d(params, ax = None, edge_color = 'darkred'):
    Q = params['Q']
    for i in range(Q):
        covariance = params[i]['cov'];
        mean = params[i]['mu'][:,0];

        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
    
        # Draw the Ellipse
        for nsig in [0.5,1,1.5,2]:
            p = Ellipse(mean, nsig*width, nsig*height, angle,fc = "none", ec = edge_color)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")       


def contour_plot_3d(params, limits, offset = 1, num_classes = None):
    """
    params: {class: {components: {'mu', 'std', 'w'}}}
    limits: [(x_min, x_max), (y_min, y_max)]
    """
    if num_classes == None:
        num_classes = len([i for i in params.keys() if type(i)==int or type(i)==float])
    for i in range(num_classes): # for each class
        # data
        x = np.linspace(limits[0][0], limits[0][1], 500)
        y = np.linspace(limits[1][0], limits[1][1],500)
        X, Y = np.meshgrid(x,y)

        # multivariate Normal
        pd = distribution(params, X, Y)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Y, pd + offset, cmap='viridis', linewidth=0)
        ax.scatter([0], [0], s= 1)
        
        # add ellipses
        draw_ellipse_3d(params, ax)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('p')

        plt.show()




def TimeGMM(X, Timepoints, Q,case = "full",tol = 1e-8):
  """
  Trains/Fits GMM Model for each class using Expectation Maximization Algorithm
  X - {t: data - (numpy 2d array)}
  Timepoints - Array of timepoints [0, 2, 4, 12, 24]
  Q - Number of clusters for each class

  returns - final parameter set : params, predicted data classes, Posterior
  """
  classes = np.array([0])
  y = np.array([0 for i in range(len(X))])

  n_t = {}; d_t = {}
  for t in Timepoints:
    n,d = np.shape(X[t])
    n_t[t] = n
    d_t[t] = d
  #n,d = np.shape(X)

  #Stores parameters (w, v, cov, c) for each class c and each subclass q
  params = {}

  #Class Prior Probabilities
  class_support = np.array([len(y[y==c])/n for c in classes]).reshape(1,-1)
  labels = {i:c for i,c in zip(range(len(classes)),classes)}

  for i in labels:
    c = labels[i]
    ite = 0

    params = {}

    log_prob_a = 0  # log likelihood before update
    log_prob_b = 0  # log likelihood after update

    #X_c = X[(y==c).ravel(),:] # Data belonging to class c
    #n_c = X_c.shape[0]  
    c_mat = np.zeros((d,Q))

    #while (abs(log_prob_a-log_prob_b)>tol or ite == 0) and ite<100:
    while ite<2:
      #*** Initialization step ***
      if ite == 0:
        time0_params = gmm(X[Timepoints[0]], Q, case) # Data of timepoint 0
        #for each component
        for q in range(Q):
          params[q] = {}
          params[q]['c'] = time0_params[q]['mu'][:,0]
          c_mat[:,q] = params[q]['c']
          params[q][Timepoints[0]] = {}
          params[q][Timepoints[0]]['cov'] = time0_params[q]['cov']
          params[q][Timepoints[0]]['w'] = time0_params[q]['w']

      #***Expectation Step - Calculating Posterior***

      #For each timepoint
      posterior = {}
      for t in range(1,len(Timepoints)):
        posterior[Timepoints[t]] = np.zeros((n_t[Timepoints[t]], Q))
        log_prob_b = log_prob_a
        log_prob_a = 0


        #For each Component
        for q in range(Q):
          if ite == 0:
            #Initializing parameters for each class using kmeans
            v_q = np.zeros((d_t[Timepoints[t]]))
            mu_q = v_q * Timepoints[t] + params[q]['c']
            #mu_q = init_params[q]['mu']
            params[q][Timepoints[t]] = {}
            params[q][Timepoints[t]]['cov'] = cov_q = params[q][Timepoints[t-1]]['cov']
            params[q][Timepoints[t]]['w'] = w_q = params[q][Timepoints[t-1]]['w']

          else:
            v_q = params[q]['v']
            cov_q = params[q][Timepoints[t]]['cov']
            w_q = params[q][Timepoints[t]]['w']

          min_eig = 1
          det =  np.linalg.det(cov_q)
          #Inspecting the Covariance Matrix and enforcing non-singularity and positive definitiveness
          while abs(det)<1e-12 or det<0 or min_eig < 0:
            cov_q += abs(np.diag((np.random.normal(0,0.5,(d,1))).ravel()))  ##Addition of Gaussian Noise
            det = np.linalg.det(cov_q)
            if det > 0:
              min_eig = np.min(np.real(np.linalg.eigvals(cov_q)))         ##Minimum Eigenvalue
          
          icov_q = np.linalg.inv(cov_q)  ##Inverse of Covariance Matrix
          #Calculating likelihood given component/cluster
          #X[t] -> (nxd), mu_q -> (dx1), cov_q -> (dxd), finally, tmp -> (nx1)
          tmp  = np.sum(np.dot((X[Timepoints[t]].T-mu_q.reshape(-1,1)).T,icov_q) * (X[Timepoints[t]].T-mu_q.reshape(-1,1)).T,axis=1)
          # pX_q_c (log-likelihood) -> (nx1)
          pX_q_c = (1/((2*np.pi)**(d_t[Timepoints[t]]/2))) * (np.linalg.det(cov_q)**-0.5) * np.exp(-0.5*tmp) 

          #Enforcing Bounds on the likelihood values to prevent small, large or nan values
          pX_q_c[pX_q_c<1e-8] = 1e-10
          pX_q_c[pX_q_c==np.inf] = 1
          pX_q_c[np.isnan(pX_q_c)] = 1e-10

          posterior[Timepoints[t]][:,q] = w_q*pX_q_c
          log_prob_a += w_q*pX_q_c ##(nx1 + ... q Q times) 

        #Calculating Total Log Probability
        log_prob_a = np.sum(np.log(log_prob_a))  ##(nx1)

        #Calculating Posterior gamma -> (nxQ)
        posterior[Timepoints[t]] = posterior[Timepoints[t]] / np.sum(posterior[Timepoints[t]],axis=1).reshape(-1,1)

      #***Maximization Step - Updating Parameters***
      #w_all -> (1xQ)
      v_num1 = np.zeros((Q, d_t[Timepoints[t]]))
      v_num2 = np.zeros((Q, d_t[Timepoints[t]]))
      v_den = np.zeros((Q, d_t[Timepoints[t]]))
      for t in range(1,len(Timepoints)):
        w_all = np.sum(posterior[Timepoints[t]],axis = 0)/n_t[Timepoints[t]]
        
        # print(w_all)
        #mu_all -> (dxQ)
        #mu_all = (1/(w_all*n_t[Timepoints[t]]))*(np.matmul(posterior[Timepoints[t]].T,X[Timepoints[t]])).T
        #print(mu_all)

        # v_num1 -> q x d
        #print(w_all)
        v_num1 += (1/(w_all.reshape(-1,1)*n_t[Timepoints[t]])) * (np.matmul(posterior[Timepoints[t]].T,X[Timepoints[t]]))
        v_num2 += (1/(w_all.reshape(-1,1)*n_t[Timepoints[t]])) * (np.sum(posterior[Timepoints[t]], axis = 0).reshape(-1,1) * c_mat.T)
        v_den  += (Timepoints[t]/(w_all.reshape(-1,1)*n_t[Timepoints[t]])) * np.sum(posterior[Timepoints[t]], axis = 0).reshape(-1,1) * (np.zeros((Q, d_t[Timepoints[t]]))+ 1)
      v_updated = (1/v_den) * (v_num1 - v_num2)
      #print(v_num1, v_num2, v_den, w_all, sep= "|")

      for t in range(1, len(Timepoints)):
        for q in range(Q):
          #params[q] = {}
          #params[q][Timepoints[t]] = {}
          params[q][Timepoints[t]]['w'] = w_all[q]
          params[q]['v'] = v_updated[q]
          #params[q]['c'] = c_mat[:,q]
          mu_q = v_updated[q]*Timepoints[t] + params[q]['c']
          #tmp -> (dxd)
          tmp  = np.matmul(np.matmul(X[Timepoints[t]].T-mu_q.reshape(-1,1),np.diag(posterior[Timepoints[t]][:,q])),(X[Timepoints[t]].T-mu_q.reshape(-1,1)).T)
          if case == "diag":
            tmp = np.diag(np.diag(tmp))

          tmp2  = w_all[q]*n_t[Timepoints[t]]

          params[q][Timepoints[t]]['cov'] = tmp/(tmp2)
      ite += 1 
        #print(params)
  params['class_support'] = class_support
  params['Q'] = Q

  #Predicting Training Labels and Class-Conditional Posterior (without dividing by Evidence) with Training Set
  #pred_classes,class_post = gmm_classifier_predict(X,classes,params)
  return params