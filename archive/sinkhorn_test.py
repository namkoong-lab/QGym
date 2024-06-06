import argparse
parser = argparse.ArgumentParser(description='Test Sinkhorn')
parser.add_argument('dim', type=int, help='Dimension of the square matrix')

args = parser.parse_args()
dim = args.dim



import torch


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import math
import torch.optim as optim
import orig_diff_des as des

from tqdm import trange
import routing as rt


from multiprocessing import Pool
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


num_iter_list = []
temp_list = []
epoch_list = []
dim_list = []
final_loss_list = []

#get args

s = dim
q = dim

mu = torch.ones((1, s, q))

# number of units to match
s_ = torch.ones(s).unsqueeze(0)
q_ = torch.ones(q).unsqueeze(0)

true_v = torch.randn((1, s, q))

pad = rt.PadValues(mu, batch = 1, f_all_vals=True)

v_input, s_bar, q_bar = pad(true_v, s_, q_)

match_true = rt.max_weight_batch(v_input, s_bar, q_bar)

v_guess = torch.randn((1, s, q))

v_guess.requires_grad = True

v_guess_input, s_bar, q_bar = pad(v_guess, s_, q_)

# match_guess = rt.Sinkhorn.apply(-v_guess_input, s_bar, q_bar, 30, 0.01, 1e-4)[:,:s,:q]

mse = nn.MSELoss()


n_trials = 5
initial_v_guesses = [torch.randn((1, s, q)) for i in range(n_trials)]

def worker(args):
    epochs_trials = []
    final_loss_trials = []

    for i_trial in range(n_trials):
        num_iter, temp, dim, s, q, initial_v_guesses, s_, q_, match_true = args
        initial_v_guess = initial_v_guesses[i_trial].clone()

        v_guess = initial_v_guess.clone()
        v_guess.requires_grad = True

        optimizer = optim.Adam([v_guess], lr = 0.1)

        epoch = 0

        prev_loss = 100000

        losses = []
        matches = []

        while True:
            # print(1)
            epoch += 1
            optimizer.zero_grad()

            v_guess_input, s_bar, q_bar = pad(v_guess, s_, q_)
            match_guess = rt.Sinkhorn.apply(-v_guess_input, s_bar, q_bar, num_iter, temp, 1e-4)[:,:s,:q]
            loss = mse(match_true, match_guess)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # print(epoch, np.abs(loss.item() - np.mean(losses[-3:])))



            # if len(losses) > 1:
            #     print(loss.item() / losses[1])
            # if len(losses) > 1 and (loss.item() / losses[1]) < 0.2:
            # if len(losses) > 3:
            #     print(loss.item(), np.abs(loss.item() - np.mean(losses[-3:])))
            matches.append((((match_guess.round() == match_true).sum()) == dim * dim).item())
            if len(matches) > 10 and np.mean(matches[-10:]) > 0.75:
                break
            
            # prev_loss = loss.item()

            if epoch > 10000:
                break

        print("dim: ", dim, "num_iter: ", num_iter, "temp: ", temp, "epoch: ", epoch, "loss: ", loss)

        epochs_trials.append(epoch)
        final_loss_trials.append(loss.item())

    return num_iter, temp, np.mean(epochs_trials), dim, np.mean(final_loss_trials)

# Now let's create the argument list
args_list = [(num_iter, temp, dim, s, q, initial_v_guesses, s_, q_, match_true) 
            for num_iter in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            for temp in [0.001, 0.01, 0.1, 1.0, 10.0]]

# Let's use multiprocessing Pool to parallelize

with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
    results = list(executor.map(worker, args_list))

# with Pool(mp.cpu_count() * 8) as pool:
# # with Pool(1) as pool:
#     results = pool.map(worker, args_list)

# Unpack results
num_iter_list_dim, temp_list_dim, epoch_list_dim, dim_list_dim, loss_list = zip(*results)

num_iter_list += num_iter_list_dim
temp_list += temp_list_dim
epoch_list += epoch_list_dim
dim_list += dim_list_dim
final_loss_list += loss_list

import pandas as pd

result = pd.DataFrame({'num_iter': num_iter_list, 'temp': temp_list, 'epoch': epoch_list, 'dim': dim_list, 'loss': loss_list})
result.to_csv(f'sinkhorn_test_dim{dim}.csv', index=False)


            




        
        
