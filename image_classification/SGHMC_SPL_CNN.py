import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List

class SGHMC_SPL(Optimizer):
    
        def __init__(self, params,N,betas=(0.1,1),eps1=5e-4,eps0=1e-3,soft_threshold=1e-4,hard_threshold=0.01,warm_up=300):
            
            if not 0.0 <= betas[0] <= 1.0:
                raise ValueError("Invalid beta1 parameter at index 0: {}".format(betas[0]))
            if not 0.0 < betas[1] <= 1.0:
                raise ValueError("Invalid beta2 parameter at index 0: {}".format(betas[1]))
            if not 0.0 <= eps1:
                raise ValueError("Invalid epsilon value: {}".format(eps1))
            if not 0.0 <= eps0:
                raise ValueError("Invalid epsilon value: {}".format(eps0))
            if not 0.0 < N:
                raise ValueError("Invalid learning rate: {}".format(N))
            if not 0.0 <= soft_threshold:
                raise ValueError("Invalid epsilon value: {}".format(soft_threshold))
            if not 0.0 <= hard_threshold:
                raise ValueError("Invalid epsilon value: {}".format(hard_threshold))

            defaults = dict(N=N,betas=betas,eps1=eps1,eps0=eps0,soft_threshold=soft_threshold,hard_threshold=hard_threshold,warm_up=warm_up)
            super(SGHMC_SPL, self).__init__(params, defaults)
        
        def __setstate__(self, state):
            super().__setstate__(state)
            state_values = list(self.state.values())
            step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
            if not step_is_tensor:
                for s in state_values:
                    s['step'] = torch.tensor(float(s['step']))
            
        @torch.no_grad()
        def step(self,C,epoch,batch,T=1/50000,eta=0.1,closure=None):
            """Performs a single SGHMC step.
            Args:
                dt: step size of Symplectic Euler Langevin scheme
                k : temperature
                eta: step size of update preconditioner parameters
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
                    
            for group in self.param_groups:
                
                params_with_grad = []
                grads = []
                rhos=[]
                taus=[]
                momentums=[]
                noises=[]
                soft_masks=[]
                hard_masks=[]
                state_steps = []
                beta1, beta2 = group['betas']
                eps1=group['eps1']
                eps0=group['eps0']
                N=group['N']
                soft_threshold=group['soft_threshold']
                hard_threshold=group['hard_threshold']
                warm_up=group['warm_up']
 
                for p in group['params']:

                    if p.grad is not None:
                        params_with_grad.append(p)
                        if p.grad.is_sparse:
                            raise RuntimeError('SGHMC does not support sparse gradients')
                        grads.append(p.grad)

                        state = self.state[p]
                        # Lazy state initialization
                        if len(state) == 0:
                            # Step
                            state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device)
                            # soft_mask
                            state['soft_mask']=torch.ones_like(p, memory_format=torch.preserve_format)
                            # hard_mask
                            state['hard_mask']=torch.ones_like(p, memory_format=torch.preserve_format)
                            # dual space of preconditioner parameter
                            state['rho'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            # preconditioner parameter
                            state['tau'] = 0.5*torch.ones_like(p, memory_format=torch.preserve_format)
                            # momentums
                            state['momentum']=0.001*torch.randn_like(p, memory_format=torch.preserve_format)
                            # noise
                            state['noise']=torch.randn_like(p, memory_format=torch.preserve_format)
                            

                        soft_masks.append(state['soft_mask'])
                        hard_masks.append(state['hard_mask'])    
                        rhos.append(state['rho'])
                        taus.append(state['tau'])
                        momentums.append(state['momentum'])
                        state_steps.append(state['step'])
                        noises.append(state['noise'])
                        
                sghmc(params_with_grad,grads,soft_masks,hard_masks,
                rhos,taus,momentums,noises,
                state_steps,beta1,beta2,
                eps1,eps0,N,warm_up,
                soft_threshold,hard_threshold,
                C=C,epoch=epoch,batch=batch,T=T,eta=eta)

            return loss

def sghmc(params: List[Tensor], grads: List[Tensor],soft_masks:List[Tensor],hard_masks:List[Tensor],
         rhos: List[Tensor],taus: List[Tensor],momentums: List[Tensor],noises: List[Tensor],
         state_steps: List[Tensor],beta1: float,beta2: float,
         eps1: float,eps0: float,N: int, warm_up:int,
         soft_threshold:float,hard_threshold:float,
         C: float,epoch:int,batch:int,T:float,eta: float):
    """Functional API that performs Sparse_EM_SGHMC algorithm.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    for i, param in enumerate(params):

        grad = grads[i] 
        rho = rhos[i]
        tau = taus[i]
        momentum=momentums[i]
        step_t = state_steps[i]
        noise=noises[i]
        soft_mask=soft_masks[i]
        hard_mask=hard_masks[i]

        eps=eps1*torch.ones_like(param, memory_format=torch.preserve_format)
        
        #EM update and pruning
        if epoch>warm_up and batch==0 and len(param.size())==4:
            ink=(param.pow(2).mean((-2,-1))<soft_threshold)
            soft_mask[ink]=0
            soft_mask[~ink]=1
            hard_mask[:,((param.amax([0,2,3])-param.amin([0,2,3]))<hard_threshold),:,:]=0
            hard_mask[((param.amax([1,2,3])-param.amin([1,2,3]))<hard_threshold),:,:,:]=0
            
        eps[soft_mask==0]=eps0

        # update step
        step_t += 1

        # Update rho
        denom1=(T**2)*(C**3)*eps*tau+T*(C**1.5)*noise*grad-torch.reciprocal(tau*N)
        rho.add_(denom1, alpha=-eta)
       
        #Update tau 
        tau.mul_(0).add_(rho.sigmoid())

        #Sample noise
        noise.normal_(0,1)

        denom2 = grad+eps*param

        #Update the momentum
        momentum.mul_(beta1).add_(denom2, alpha=-C**2*beta2).add_(noise,alpha=((2+2*beta1**2))**0.5*(C**1.5)*T)

        #Update the parameter
        param.addcmul_(tau, momentum, value=1).mul_(hard_mask)
      
     