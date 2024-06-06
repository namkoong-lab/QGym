import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import torch
from torch import nn
import torch.nn.functional as F

# preemptive policy

def match_constraint_mat(num_s, num_q, f_fluid = False):
    #rhs = np.append(data['s'],-data['d'])

    # form A
    # column (i,j)=n*i+j has two nonzeroes:
    #    1 at row i with rhs supply(i)
    #    1 at row N+j with rhs demand(j)
    N = num_s
    M = num_q
    NZ = 2*N*M
    irow = np.zeros(NZ, dtype=int)
    jcol = np.zeros(NZ, dtype=int)
    value = np.zeros(NZ)
    for i in range(N):
        for j in range(M):
            k = M*i+j
            k1 = 2*k
            k2 = k1+1
            irow[k1] = i
            jcol[k1] = k
            value[k1] = 1.0
            if not f_fluid:
                irow[k2] = N+j
                jcol[k2] = k
                value[k2] = 1.0

    A = sparse.coo_matrix((value, (irow, jcol)))
    
    return A


def linear_assignment(values, servers, jobs):

    s,q = values.shape
    A = match_constraint_mat(s, q).toarray()
    c = np.reshape(-values, s * q)
    b = np.append(servers, jobs)

    res = opt.linprog(c=c,A_ub=A,b_ub=b, method = 'highs-ds')
    X = np.reshape(res.x, (s,q))

    X = np.rint(X)[:s-1,:q-1].tolist()
        
    return torch.tensor(X)


def linear_assignment_batch(values, s_bar, q_bar):

    batch,s,q = values.size()
    action = []

    for b in range(batch):
        v = values[b].numpy()
        servers = s_bar[b].numpy()
        jobs = q_bar[b].numpy()

        A = match_constraint_mat(s, q).toarray()
        c = np.reshape(-v, s * q)
        b = np.append(servers, jobs)

        res = opt.linprog(c=c,A_ub=A,b_ub=b, method = 'highs-ds')
        X = np.reshape(res.x, (s,q))

        X = np.rint(X)[:s-1,:q-1].tolist()
        action.append(X)
    
    return torch.tensor(action)


def pad(vals, queues, network, 
        device = 'cpu', compliance = True):

    # setup mu bar
    batch = network.size()[0]
    s = network.size()[1]
    q = network.size()[2]

    free_servers = torch.ones((batch, s)).to(device)
    
    # pad_q = torch.zeros((batch, 1,q)).to(device)
    # pad_s = torch.zeros((batch, s + 1,1)).to(device)
    pad_q = -torch.ones((batch, 1,q)).to(device)
    pad_s = -torch.ones((batch, s + 1,1)).to(device)

    if compliance:
        vals = vals * network - 1*(network == 0.).to(device)

    v = torch.cat((vals, pad_q), 1)
    v = torch.cat((v, pad_s), 2)

    excess_server = F.relu(s - torch.sum(queues, dim = 1)).unsqueeze(1).to(device)
    q_bar = torch.hstack((queues, excess_server)).to(device)

    excess_queues = F.relu(torch.sum(queues, dim = 1) - s).unsqueeze(1).to(device)
    s_bar = torch.hstack((free_servers, excess_queues)).to(device)

    return v, s_bar, q_bar

def pad_pool(vals, queues, network, server_pool_size,
        device = 'cpu', compliance = True):

    
    # setup mu bar
    batch = network.size()[0]
    s = server_pool_size.sum()
    q = network.size()[2]

    free_servers = server_pool_size.repeat(batch, 1).to(device)
    # torch.ones((batch, network.shape[1])).to(device)
    
    # pad_q = torch.zeros((batch, 1,q)).to(device)
    # pad_s = torch.zeros((batch, s + 1,1)).to(device)
    pad_q = -torch.ones((batch, 1,q)).to(device)
    pad_s = -torch.ones((batch, network.shape[1] + 1,1)).to(device)

    if compliance:
        vals = vals * network - 1*(network == 0.).to(device)

    # if len(vals.shape) == 4:
    #     vals = vals[0]
    # print(vals.shape)
    # print(pad_q.shape)
    v = torch.cat((vals, pad_q), 1)
    # print(v.shape, pad_s.shape)
    v = torch.cat((v, pad_s), 2)

    excess_server = F.relu(s - torch.sum(queues, dim = 1)).unsqueeze(1).to(device)
    q_bar = torch.hstack((queues, excess_server)).to(device)

    excess_queues = F.relu(torch.sum(queues, dim = 1) - s).unsqueeze(1).to(device)
    s_bar = torch.hstack((free_servers, excess_queues)).to(device)

    return v, s_bar, q_bar

# class PadValues(nn.Module):
#     def __init__(self, mu, batch, device = "cpu", f_all_vals = False, f_norm = False):
#         super().__init__()
#         self.device = device
#         self.mu = mu.to(self.device)
#         self.batch = batch
#         self.f_all_vals = f_all_vals
#         self.f_norm = f_norm
#         self.s = self.mu.size()[1]
#         self.q = self.mu.size()[2]
    
#         # setup mu bar
#         self.pad_q = torch.zeros((self.batch, 1,self.q)).to(self.device)
#         self.pad_s = torch.zeros((self.batch, self.s + 1,1)).to(self.device)
        
#         self.mu_bar = torch.cat((self.mu, self.pad_q), 1)
#         self.mu_bar = torch.cat((self.mu_bar, self.pad_s), 2)
#         self.network = 1*(self.mu_bar > 0).to(self.device)
#         self.anti_network = 1*(self.mu_bar == 0).to(self.device)
        
#     def forward(self, vals, free_servers, queues):
        
#         # pad the value matrix
#         if self.f_all_vals:
#             v = torch.cat((vals, self.pad_q), 1)
#             v = torch.cat((v, self.pad_s), 2)
#         else:
#             pad_values = torch.zeros((self.batch, 1)).to(self.device)
#             v = torch.cat((vals, pad_values), 1).unsqueeze(1).to(self.device)
#             v = (v * self.mu_bar)
            
#         # multiply by mu matrix
#         #vals = (v * self.mu_bar)
#         vals = (v * self.network)

#         # normalize and scale
#         if self.f_norm:
#             # min{1/std, 1}
#             #scale = torch.minimum(1/torch.std(v, (1,2)), torch.tensor([1.])).unsqueeze(1).unsqueeze(2)
#             var = torch.var(v, 2).reshape(self.batch, self.s + 1, 1).to(self.device)
#             mean = torch.mean(v, 2).reshape(self.batch, self.s + 1, 1).to(self.device)
#             scale = torch.minimum(1/torch.sqrt(var + 0.01), torch.tensor([1.])).to(self.device)

#             # var = torch.var(v, (1,2)).unsqueeze(1).unsqueeze(2)
#             # mean = torch.mean(v, (1,2)).unsqueeze(1).unsqueeze(2)
#             # scale = torch.minimum(1/torch.sqrt(var + 0.01), torch.tensor([2.]))
#             vals = (v - mean) * scale

#         # Penalize non-valid entries
#         v = vals - self.anti_network
        
#         # excess servers and jobs
#         excess_server = F.relu(self.s - torch.sum(queues, dim = 1)).unsqueeze(1).to(self.device)
#         q_bar = torch.hstack((queues, excess_server)).to(self.device)

#         excess_queues = F.relu(torch.sum(queues, dim = 1) - self.s).unsqueeze(1).to(self.device)
#         s_bar = torch.hstack((free_servers, excess_queues)).to(self.device)
        
#         return v, s_bar, q_bar
        
class Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, a, b, num_iter, temp, eps = 1e-6, back_temp = None, device = 'cpu'):
        
        log_p = -c / temp
        
        a_dim = 2
        b_dim = 1

        log_a = torch.log(torch.clamp(a, eps)).unsqueeze(dim=2)
        log_b = torch.log(torch.clamp(b, eps)).unsqueeze(dim=1)

        for _ in range(num_iter):
            log_p -= (torch.logsumexp(log_p, dim=1, keepdim=True) - log_b)
            log_p -= (torch.logsumexp(log_p, dim=2, keepdim=True) - log_a)
        
        p = torch.exp(log_p)
        ctx.save_for_backward(p, torch.sum(p, dim=2), torch.sum(p, dim=1))
        ctx.temp = temp
        ctx.back_temp = back_temp
        ctx.device = device
        
        return p

    @staticmethod
    def backward(ctx, grad_p):
        
        p, a, b = ctx.saved_tensors
        batch, m, n = p.shape

        device = ctx.device
        
        a = torch.clamp(a, 1e-1)
        b = torch.clamp(b, 1e-1)
        
        if ctx.back_temp is not None:
            grad_p *= -1 / ctx.back_temp * p
        else:
            grad_p *= -1 / ctx.temp * p

        K_b = torch.cat((
            torch.cat((torch.diag_embed(a), p), dim=2),
            torch.cat((torch.transpose(p, 1, 2), torch.diag_embed(b)), dim=2)),
            dim = 1)[:,:-1,:-1]
        
        I = torch.eye(K_b.size()[1]).to(device)
        n_batch = torch.tensor([1.0]*batch).to(device)
        batch_eye = torch.einsum('ij,k->kij', I, n_batch)
        
        K_b = K_b + 0.01*batch_eye


        t_b = torch.cat((
            grad_p.sum(dim=2),
            grad_p[:,:,:-1].sum(dim=1)),
            dim = 1).unsqueeze(2)


        grad_ab_b = torch.linalg.solve(K_b, t_b)
        grad_a_b = grad_ab_b[:, :m, :]
        grad_b_b = torch.cat((grad_ab_b[:, m:, :], torch.zeros((batch, 1, 1), dtype=torch.float32).to(device)), dim=1)

        U = grad_a_b + torch.transpose(grad_b_b, 1, 2)

        grad_p -= p * U
#         grad_a = -ctx.temp * grad_a.squeeze(dim=1)
#         grad_b = -ctx.temp * grad_b.squeeze(dim=1)
        
        return grad_p, None, None, None, None, None, None, None

# class NoBatchSinkhorn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, c, a, b, num_sink, lambd_sink, eps = 1e-5):
#         log_p = -c / lambd_sink
#         log_a = torch.log(torch.clamp(a, eps)).unsqueeze(dim=1)
#         log_b = torch.log(torch.clamp(b, eps)).unsqueeze(dim=0)
#         for _ in range(num_sink):
#             log_p -= (torch.logsumexp(log_p, dim=0, keepdim=True) - log_b)
#             log_p -= (torch.logsumexp(log_p, dim=1, keepdim=True) - log_a)
#         p = torch.exp(log_p)
#         ctx.save_for_backward(p, torch.sum(p, dim=1), torch.sum(p, dim=0))
#         ctx.lambd_sink = lambd_sink
#         return p
#     @staticmethod
#     def backward(ctx, grad_p):
#         p, a, b = ctx.saved_tensors
#         m, n = p.shape
        
#         lambd_sink = ctx.lambd_sink
#         grad_p *= -1 / lambd_sink * p
#         K = torch.cat((
#         torch.cat((torch.diag(a), p), dim=1),
#         torch.cat((p.T, torch.diag(b)), dim=1)),
#         dim=0)[:-1, :-1]
#         t = torch.cat((
#         grad_p.sum(dim=1),
#         grad_p[:, :-1].sum(dim=0)),
#         dim=0).unsqueeze(1)
#         grad_ab = torch.linalg.solve(K,t)
#         grad_a = grad_ab[:m, :]
#         grad_b = torch.cat((grad_ab[m:, :], torch.zeros([1, 1],
#         device='cpu', dtype=torch.float32)), dim=0)
#         U = grad_a + grad_b.T
#         grad_p -= p * U
#         grad_a = -lambd_sink * grad_a.squeeze(dim=1)
#         grad_b = -lambd_sink * grad_b.squeeze(dim=1)
#         return grad_p, grad_a, grad_b, None, None, None
    


# class SinkhornLayer(nn.Module):
#     def __init__(self, mu, batch, temp = 1/2, num_iter = 5, device = "cpu", f_mu = True, f_hot_cold = False):
#         super().__init__()
#         self.device = device
#         self.temp = temp
#         self.mu = mu.to(self.device)
#         self.f_mu = f_mu
#         self.batch = batch
#         self.eps = 1e-3
#         self.s = self.mu.size()[1]
#         self.q = self.mu.size()[2]
#         self.num_iter = num_iter
#         self.f_hot_cold = f_hot_cold
    
#         # setup mu bar
#         pad_q = torch.zeros((self.batch, 1,self.q)).to(self.device)
#         pad_s = torch.zeros((self.batch, self.s + 1,1)).to(self.device)
        
#         self.mu_bar = torch.cat((self.mu, pad_q), 1)
#         self.mu_bar = torch.cat((self.mu_bar, pad_s), 2)
        
#     def forward(self, vals, free_servers, queues):
        
#         # pad the value matrix
#         pad_values = torch.zeros((self.batch, 1)).to(self.device)
#         v = torch.cat((vals, pad_values), 1).unsqueeze(1).to(self.device)
        
#         # multiply by mu matrix
#         v = (v * self.mu_bar)
        
#         # excess servers and jobs
#         excess_server = F.relu(self.s - torch.sum(queues, dim = 1)).unsqueeze(1).to(self.device)
#         q_bar = torch.hstack((queues, excess_server)).to(self.device)

#         excess_queues = F.relu(torch.sum(queues, dim = 1) - self.s).unsqueeze(1).to(self.device)
#         s_bar = torch.hstack((free_servers, excess_queues)).to(self.device)
        
#         # Run Sinkhorn loop
#         log_p = v / self.temp
        
#         a = s_bar
#         b = q_bar
    
#         a_dim = 2
#         b_dim = 1

#         log_a = torch.log(torch.clamp(a, self.eps)).unsqueeze(dim=a_dim)
#         log_b = torch.log(torch.clamp(b, self.eps)).unsqueeze(dim=b_dim)

#         for _ in range(self.num_iter):
#             log_p = log_p - (torch.logsumexp(log_p, dim=b_dim, keepdim=True) - log_b)
#             log_p = log_p - (torch.logsumexp(log_p, dim=a_dim, keepdim=True) - log_a)

#         p = torch.exp(log_p)[:,:self.s,:self.q]
        
#         if not self.f_hot_cold:
#             return p
#         else:
#             # Run Sinkhorn loop
#             log_p_hot = v

#             a = s_bar
#             b = q_bar

#             a_dim = 2
#             b_dim = 1

#             log_a = torch.log(torch.clamp(a, self.eps)).unsqueeze(dim=a_dim)
#             log_b = torch.log(torch.clamp(b, self.eps)).unsqueeze(dim=b_dim)

#             for _ in range(self.num_iter):
#                 log_p_hot = log_p_hot - (torch.logsumexp(log_p_hot, dim=b_dim, keepdim=True) - log_b)
#                 log_p_hot = log_p_hot - (torch.logsumexp(log_p_hot, dim=a_dim, keepdim=True) - log_a)

#             p_hot = torch.exp(log_p_hot)[:,:self.s,:self.q]
            
#             return p.detach() + p_hot - p_hot.detach()
            