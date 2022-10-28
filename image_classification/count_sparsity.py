import torch

def sparsity(net):

    total_weights = 0
    nonzero_weight=0

    for param in net.parameters():

        nonzero_weight += torch.count_nonzero(param)

        total_weights += param.numel()
    
    return (total_weights-nonzero_weight)/total_weights

    



