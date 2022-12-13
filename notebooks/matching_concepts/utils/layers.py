from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import clip
from PIL import Image
import requests
import torch.hub
import time
import pickle
import math

from utils import matching, stats, proggan, nethook, zdataset, loading, plotting






def get_layers(gan, gan_layers, clipmodel,clip_layers, device):
    gan.eval()
    clipmodel.eval()
    
    #### hook layers for GAN
    gan = nethook.InstrumentedModel(gan)
    gan.retain_layers(gan_layers)
    
    
    #### hook layers for CLIP
    clipmodel = nethook.InstrumentedModel(clipmodel)
    clipmodel.retain_layers(clip_layers)
    
    
   
        
      
    dataset = torch.rand

    z = torch.randn((1,128)).to(device).detach()
    c = torch.zeros((1,1000)).to(device).detach()    
    
        
    #### Forward through GAN
    with torch.no_grad():
        img = gan(z,c,0.5).detach()
            
        
        del z
        

        #### append GAN layer activations for batch
    gan_activs = []
    for layer in gan_layers:
        gan_activation = gan.retained_layer(layer, clear = True).detach()
        gan_activs.append(gan_activation)
                
        


        #### Prepare images for CLIP
    img = (img+1)/2
    img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
    img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
       
        
        #### Forward through CLIP
    with torch.no_grad():
        _ = clipmodel.model.encode_image(img)
        
    del img
        

        
        
        #### append CLIP layer activations for batch
    clip_activs = []
    for layer in clip_layers:
        clip_activation = clipmodel.retained_layer(layer, clear = True).detach()
        clip_activs.append(clip_activation)
        

        #create dict of layers
    all_gan_layers = {}
    for iii, gan_activ in enumerate(gan_activs):
        all_gan_layers[gan_layers[iii]] = gan_activ.shape[1]
        
    all_clip_layers = {}
    for jjj, clip_activ in enumerate(clip_activs):
        all_clip_layers[clip_layers[jjj]] = clip_activ.shape[1]
            
    return all_gan_layers, all_clip_layers







def find_act(act_num, net_dict):
    layers_list = list(net_dict)
    
    layer = 0
    counter =0
    
    while act_num >= counter:
        layer +=1
        counter += net_dict[layers_list[layer-1]]
        
        
    act = act_num-counter+net_dict[layers_list[layer-1]]
    
    del layers_list
    torch.cuda.empty_cache()
    return (layer-1), act
        