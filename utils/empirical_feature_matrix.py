from copy import deepcopy
from statistics import mode
import torch
import itertools 
import sys 
from torch import autograd
from torch import nn 
from tqdm import tqdm
import os 
from torch.autograd import grad
import numpy as np 
import einops

def jacobian_in_batch(y, x):
    '''
    Compute the Jacobian matrix in batch form.
    Return (B, D_y, D_x)
    '''

    batch = y.shape[0]
    single_y_size = np.prod(y.shape[1:])
    y = y.view(batch, -1)
    vector = torch.ones(batch).to(y)

    # Compute Jacobian row by row.
    # dy_i / dx -> dy / dx
    # (B, D) -> (B, 1, D) -> (B, D, D)
    jac = [torch.autograd.grad(y[:, i], x, 
                            grad_outputs=vector, 
                            retain_graph=True,
                            create_graph=True)[0].view(batch, -1)
                for i in range(single_y_size)]
    jac = torch.stack(jac, dim=1)
    
    return jac
    
def isPSD(A, tol=1e-7):
    import numpy as np 
    A = A.cpu().numpy()
    E = np.linalg.eigvalsh(A)
    print("Maximum eigenvalue {}".format(np.max(E)))
    return np.all(E > -tol)


class EmpiricalFeatureMatrix:


    def __init__(self,  device, out_path):
        self.empirical_feat_mat = None
        self.device = device
        self.out_path = out_path
    
 
    def create(self, model):
        self.empirical_feat_mat = torch.zeros((model.feat_dim,model.feat_dim), requires_grad=False).to(self.device)

    
    def get(self):
        return self.empirical_feat_mat


    def compute(self, model, trn_loader, concepts,protos,proto_space):
        self.compute_efm(model, trn_loader, concepts,protos,proto_space)
                

 
    def compute_efm(self, model, trn_loader, concepts,protos,proto_space):
        print("Evaluating Empirical Feature Matrix")
        # Compute empirical feature matrix for specified number of samples -- rounded to the batch size
        
        n_samples_batches = len(trn_loader.dataset) // trn_loader.batch_size

        model.eval() 
        # ensure that gradients are zero
        model.zero_grad()   

        self.create(model)
    
        with torch.no_grad():
            for _, images, targets in itertools.islice(trn_loader, n_samples_batches):

                gap_out = model.extract_vector(images.to(self.device))
                
                out = torch.cat([h(gap_out)for h in model.heads], dim=1)
                out_size = out.shape[1]
                
                # compute the efm using the closed formula    
                log_p =  nn.LogSoftmax(dim=1)(out)
                
                identity = torch.eye(out.shape[1], device=self.device)
                
                der_log_softmax =  einops.repeat(identity, 'n m -> b n m', b=gap_out.shape[0]) - einops.repeat(torch.exp(log_p), 'n p -> n a p', a=out.shape[1])
                
                weight_matrix = torch.cat([h.weight for h in model.heads], dim=0) 
                # closed formula jacobian matrix
                jac  = torch.bmm(der_log_softmax, einops.repeat(weight_matrix, 'n m -> b n m', b=gap_out.shape[0]))
                
                efm_per_batch = torch.zeros((images.shape[0], weight_matrix.shape[1], weight_matrix.shape[1]), device=self.device)
                
                p = torch.exp(log_p)
                
                # jac =  jacobian_in_batch(log_p, gap_out).detach() # equivalent formulation with gradient computation, with torch.no_grad() should be removed
            
                for c in range(out_size):
                    efm_per_batch +=  p[:,c].view(images.shape[0], 1, 1) * torch.bmm(jac[:,c, :].unsqueeze(1).permute(0,2,1), jac[:,c, :].unsqueeze(1)) 
                    
                self.empirical_feat_mat +=  torch.sum(efm_per_batch,dim=0)   
                
        
        n_samples = n_samples_batches * trn_loader.batch_size

        # divide by the total number of samples 
        self.empirical_feat_mat = self.empirical_feat_mat/n_samples

        if isPSD(self.empirical_feat_mat):
            print("EFM is semidefinite positive")

        return self.empirical_feat_mat 
          
  
    
