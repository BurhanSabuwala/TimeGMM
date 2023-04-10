# TimeGMM

```{python} 
TimeGMM(data, Timepoints, Q, case = "diag", tol = 1e-8)
```

Parameters

----------

Timepoints - [t_1, t_2, ... t_T] # Timepoints for which the data is provided in the data object 


data - {Timepoints[t]: n_txd_t size 2-D array | for t in Timepoints}


Q - Number of components


Returns

-------

Connects all time points. Trains a GMM model on the first timepoint given in the Timepoint list. Estimates the v parameter based on the 


```{python}
stepwise_TimeGMM(X, Timepoints, Q, case = "diag", tol = 1e-8)
```
Connects two immediate time points. Trains a GMM model on timepoint t and extrapolates to timepoint t+1 to estimate the v. Models stepwise progression this way.

```{python}
single_gaussian_best_fit(X, Timepoints, beta = 0.001, case = "diag", tol = 1e-8)
```
