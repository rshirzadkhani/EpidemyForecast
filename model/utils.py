import networkx as nx
import numpy as np
import math
import statsmodels.stats.api as sms
from scipy import stats

def compute_mean_confidence_interval(seir, alpha=0.95):
    # seir.shape = (num_seeds, len(npi_strength), T, len(SEIR_COMPARTMENTS))
    mean = np.average(seir[:,1:], axis=0)
    ci = np.zeros(mean.shape)
    lower = np.zeros(mean.shape)
    upper = np.zeros(mean.shape)
    # print("shapes:",mean.shape, ci.shape)
    # print(seir.shape)
    for i in range(1, seir.shape[1]-1):
        # for j in range(seir.shape[2]):
            # for k in range(seir.shape[3]):
                theta = seir[:,i+1]
                # print(theta)
                # n = len(theta)
                # se = stats.sem(theta) / math.sqrt(n)
                if seir.shape[0] < 30:
                    # ci[i,j,k] = se * stats.t.interval((1 + (1-alpha)) / 2., n-1)
                    lower[i], upper[i] = stats.t.interval(alpha=0.95, 
                                                   df=len(theta)-1,
                                                   loc=np.mean(theta), 
                                                   scale=stats.sem(theta))
                else:
                    # ci[i,j,k] = se * stats.norm.ppf(1-(alpha/2))
                    lower[i], upper[i] = stats.norm.interval(alpha=0.95,
                                                   loc=np.mean(theta),
                                                   scale=stats.sem(theta))            
    return mean, lower, upper