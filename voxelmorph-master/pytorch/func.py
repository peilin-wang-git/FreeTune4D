import numpy as np
from torch.optim import lr_scheduler

def get_empty_array(*args):
    losses = {}
    
    for key in args:
        losses[key] = np.array([])
        
    return losses

def array_append(losses, **kwargs):
    for key in kwargs.keys():
        losses[key] = np.append(losses[key], kwargs[key].to('cpu').data.item())
        
    return losses

def array_append_mean(losses, losses_temp):
    for key in losses_temp.keys():
        losses[key] = np.append(losses[key], np.mean(losses_temp[key]))    
        
    return losses
    
def get_scheduler(optimizer,decay_policy, **kwargs):

    opt = DecayOption(decay_policy)
    for i in kwargs.keys():
        setattr(opt, i, kwargs[i])
        
    if decay_policy == 'plateau_lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter_plateau) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
     
    elif decay_policy == 'MultiStepLR':
        milestone = list(map(lambda x: int(x), opt.milestone.split('-')))
        gamma = opt.gamma
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestone,gamma)

    return scheduler

def update_learning_rate(scheduler, optimizer, min_lr=1e-8):
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.12f' % lr)  
    
    if lr > min_lr:
        scheduler.step()
 
    
class DecayOption():
    def __init__(self, decay_policy):
        self.decay_policy = decay_policy
        

    



# losses = get_empty_array('ss')
# losses = array_append(losses, ss=torch.Tensor([1]).to(device))