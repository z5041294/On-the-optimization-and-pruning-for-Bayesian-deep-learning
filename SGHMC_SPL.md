CLASS SGHMC_SPL(param,N,weight_decay_1=5e-4,weight_decay_0=1e-1,soft_threshold=1e-2,hard_threshold=1e-3,warm_up=3000)





# Parameters

* **params** (*iterable*) – iterable of parameters to optimize or dicts defining parameter groups
* N -training size
* betas -momentum factor (default: $\beta_{0}=1$,$\beta_{1}$=0.1)
* weight_decay_1 - small weight decay factor(small L2 penalty)
* weight_decay_0 - large weight decay factor(large L2 penalty)

* soft_threshold  -  Threshold for EM algorithm to switch between small and large weight decay factor
* Hard_threshold - Threshold for pruning
* warm_up - it warm_up >epoch 



# Example

```python
optimizer = SGHMC_SPL(net.parameters(),N=len(train_dataloader.dataset),weight_decay_1=2e-4,weight_decay_0=1e-2,soft_threshold=1e-2,hard_threshold=1e-3,warm_up=50)


def adjust_learning_rate(epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    C2 = 0.5*cos_out*C2_0
    return C2

def train_loop(epoch,dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print('\nEpoch: %d' % (epoch+1))
    running_loss = 0.0
    
    model.train()
        
    for batch, (X, Y) in enumerate(dataloader):
        
        # Compute prediction and loss
        X=X.to(device1)
        Y=Y.to(device1)
        pred = model(X)
        loss = loss_fn(pred, Y)
        C = adjust_learning_rate(epoch+1,batch)**0.5
        eta=1
        T=(1/size)**0.5
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(C,epoch=epoch,batch=batch,T=T)

        running_loss += loss.item()

```



