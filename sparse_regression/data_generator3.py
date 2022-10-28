import numpy as np
import torch
from scipy.stats import truncnorm

class simulate_data:
    
    #constructor
    def __init__(self,N=10000,P=1000,device='cuda'):

        np.random.seed(5041294)
        error=np.random.randn(N,1)
        r = truncnorm.rvs(-10, 10, size=(N,P+1))
        X_n=(r[:,0:1]+r[:,1:])/2**0.5
        ink=np.exp(X_n[:,0:1])+X_n[:,1:2]**2+5*np.sin(X_n[:,2:3]*X_n[:,3:4])-3
        Y_n=np.zeros((N,1))
        Y_n[ink>0]=1
        self.X=torch.from_numpy(X_n).to(device).float()
        self.Y=torch.from_numpy(Y_n).to(device).float()
        
    #Get Length
    def __len__(self):
        return len(self.Y)
    
    #Getter
    def __getitem__(self, index):    
        return self.X[index], self.Y[index]    