### Libraries
import GMM 
import numpy as np
import pandas as pd



x = np.random.random(size = 1000)

print(GMM.kmeans(x.ravel(), 2))
