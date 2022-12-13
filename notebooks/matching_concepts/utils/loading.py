import torch
import pickle
import os

def load_stats(root, device):
    print("Loading...")
    file_name = os.path.join(root, "table.pkl")
    with open(file_name, 'rb') as f:
        table = pickle.load(f)
        table = table.to(device)#cpu()
    
    with open(os.path.join(root,"clip_stats.pkl"), 'rb') as f:
        clip_stats = pickle.load(f)
        
        for iii, item1 in enumerate(clip_stats):
            for jjj, item2 in enumerate(clip_stats[iii]):
                clip_stats[iii][jjj] = clip_stats[iii][jjj].to(device)
                
        
    with open(os.path.join(root,"gan_stats.pkl"), 'rb') as f:
        gan_stats = pickle.load(f)
        for iii, item1 in enumerate(gan_stats):
            for jjj, item2 in enumerate(gan_stats[iii]):
                gan_stats[iii][jjj] = gan_stats[iii][jjj].to(device)
                
        
        
    print("Done")
    return table, gan_stats, clip_stats