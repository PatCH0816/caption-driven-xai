import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_stats(gan_stats, clip_stats, table):
    
    gan_means = []
    gan_stds = []
    for iii, layer in enumerate(gan_stats):
        gan_means.append(gan_stats[iii][0].flatten().unsqueeze(0))
        gan_stds.append(gan_stats[iii][1].flatten().unsqueeze(0))

    gan_means = torch.cat(gan_means,1)
    gan_stds = torch.cat(gan_stds,1)
    
    
    
    
    
    clip_means = []
    clip_stds = []
    for iii, layer in enumerate(clip_stats):
        clip_means.append(clip_stats[iii][0].flatten().unsqueeze(0))
        clip_stds.append(clip_stats[iii][1].flatten().unsqueeze(0))

    clip_means = torch.cat(clip_means,1)
    clip_stds = torch.cat(clip_stds,1)
    
    
    
    
    ### scores
    
    table_flattened = table.flatten()
    scores, flat_indices = torch.sort(table_flattened, descending = True)
    flat_indices_matches = flat_indices[0:10000].cpu()
    indices_matches = np.unravel_index(flat_indices_matches, (table.shape[0], table.shape[1]))
    
    

    
    
    #fig=plt.figure(figsize=(15, 15))
    #plt.title('Activation Statistics')
    #plt.axis('off')
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(clip_means.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("CLIP Means")
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(clip_stds.cpu(), bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5,1, 5])
    plt.title("CLIP STDs")
   # plt.xticks(np.arange(0, 1, 0.1), size =5)
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan_means.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("GAN Means")
    plt.show()
    
    
    fig, ax = plt.subplots(figsize =(3, 3))
    ax.hist(gan_stds.cpu(), bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20])
    plt.title("GAN STDs")
    plt.show()
    
    
    
    
    gan_score_idxs,_ = torch.max(table, dim = 1)
    fig, ax = plt.subplots(figsize =(10,6))
    ax.plot(gan_score_idxs.cpu())
    plt.title("Scores vs. GAN Depth")
    plt.show()
    
    
    
    clip_score_idxs,_ = torch.max(table, dim = 0)
    fig, ax = plt.subplots(figsize =(10,6))
    ax.plot(clip_score_idxs.cpu())
    plt.title("Scores vs. CLIP Depth")
    plt.show()