from utils import nethook, zdataset, stats, helpers

import torch
import torchvision
import numpy as np
import pickle
import os





def normalize(activation, stats_table):
    eps = 0.00001
    norm_input = (activation- stats_table[0])/(stats_table[1]+eps)
    
    return norm_input

def store_activs(model, layers):
    activs = []
    for layer in layers:
        activation = model.retained_layer(layer, clear = True)
        activs.append(activation)
        
    return activs

def dict_layers(activs):
    all_layers = {}
    for iii, activ in enumerate(activs):
        all_layers[activs[iii]] = activ.shape[1]
    return all_layers
    


def activ_match_gan(gan, gan_layers, clipmodel,clip_layers, dataset, epochs, batch_size,save_path, device):
    gan.eval()
    clipmodel.eval()
    
    #### hook layers for GAN
    gan = nethook.InstrumentedModel(gan)
    gan.retain_layers(gan_layers)
    
    #### hook layers for CLIP
    clipmodel = nethook.InstrumentedModel(clipmodel)
    clipmodel.retain_layers(clip_layers)
    
    #get dataset stats
    gan_stats_table, clip_stats_table = stats.get_mean_std(gan, gan_layers, clipmodel, clip_layers, dataset, epochs, batch_size, device)
    helpers.save_array(gan_stats_table, os.path.join(save_path, "gan_stats.pkl"))
    helpers.save_array(clip_stats_table, os.path.join(save_path, "clip_stats.pkl"))
    
    print("Done")
    print("Starting Activation Matching")
    
    

    for iteration in range(0, epochs):
        with torch.no_grad():
            print("Iteration: "+ str(iteration))
            #### dataset
            z = dataset[0][iteration*batch_size: (iteration+1)*batch_size ]
            c = dataset[1][iteration*batch_size: (iteration+1)*batch_size ]
            #### Forward through GAN
            img = gan(z,c,1)
            del z
            #### append GAN layer activations for batch
            gan_activs = store_activs(gan, gan_layers)

            #### Prepare images for CLIP
            img = (img+1)/2
            img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
            img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)

            #### Forward through CLIP
            _ = clipmodel.model.encode_image(img)
            del img


            #### append CLIP layer activations for batch
            clip_activs =  store_activs(clipmodel, clip_layers)

            #create dict of layers with number of activations
            all_gan_layers = dict_layers(gan_activs)
            all_clip_layers = dict_layers(clip_activs)
            
            
            if iteration == 0:
                num_gan_activs = sum(all_gan_layers.values())
                num_clip_activs = sum(all_clip_layers.values())
                final_match_table = torch.zeros((num_gan_activs, num_clip_activs)).to(device)



            ##### Matching
            all_match_table = []

            for ii, gan_activ in enumerate(gan_activs):
                match_table = []
                gan_activ = normalize(gan_activ, gan_stats_table[ii])
                gan_activ_shape = gan_activ.shape

                for jj, clip_activ in enumerate(clip_activs):
                    clip_activ_new = normalize(clip_activ, clip_stats_table[jj]) 
                    #### Get maps to same size
                    map_size = max((gan_activ_shape[2], clip_activ.shape[2]))
                    gan_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan_activ)
                    clip_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(clip_activ_new)            
                    scores = torch.einsum('aixy,ajxy->ij', gan_activ_new,clip_activ_new)/(batch_size*map_size**2)  
                    scores = scores.cpu()
                    
                    match_table.append(scores)
                    del gan_activ_new
                    del clip_activ_new
                    del scores

                all_match_table.append(match_table)
                del match_table


            ##create table
            batch_match_table = create_final_table(all_match_table, all_gan_layers, all_clip_layers, batch_size, device)
            final_match_table += batch_match_table
            #helpers.save_array(final_match_table, os.path.join(save_path, "norm_table_"+str(iteration)+".pkl"))
        
            del all_match_table
            del batch_match_table
            del gan_activs
            del clip_activs
            torch.cuda.empty_cache()
            
    #average and save
    final_match_table /= epochs
    helpers.save_array(final_match_table, os.path.join(save_path, "table.pkl"))
    
    
    
def create_final_table(all_match_table, model1_dict, model2_dict, batch_size, device ):
    num_activs1 = sum(model1_dict.values())
    num_activs2 = sum(model2_dict.values())
    final_match_table = torch.zeros((num_activs1, num_activs2)).to(device)
    
    model1_activ_count = 0 
    for ii in range(len(all_match_table)):
        model2_activ_count = 0
        for jj in range(len(all_match_table[ii])):
            num_model1activs = all_match_table[ii][0].shape[0]
            num_model2activs = all_match_table[0][jj].shape[1]
            final_match_table[model1_activ_count: model1_activ_count+num_model1activs, \
                            model2_activ_count:model2_activ_count+num_model2activs] = all_match_table[ii][jj]
            model2_activ_count += num_model2activs
        model1_activ_count += num_model1activs
    return final_match_table
    
    
    
    
    
    
    
    
    
    
    