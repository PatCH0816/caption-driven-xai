from utils import nethook, zdataset
import torch
import torchvision






invTransGAN = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [1/0.5, 1/0.5, 1/0.5]),
                                torchvision.transforms.Normalize(mean = [-0.5, -0.5, -0.5],
                                                     std = [ 1., 1., 1. ]),
                               ]) 







def get_mean_std(gan, gan_layers, clipmodel, clip_layers, dataset, epochs, batch_size, device):
    print("Collecting Dataset Statistics")
    
    gan_stats_list = []
    clip_stats_list = []
    with torch.no_grad():
        for iteration in range(0, epochs):
            print(iteration)
            z = dataset[0][iteration*batch_size: (iteration+1)*batch_size ]
            c = dataset[1][iteration*batch_size: (iteration+1)*batch_size ]

            img = gan(z,c,1)

            #### append GAN layer activations for batch
            gan_activs = {}
            for layer in gan_layers:
                gan_activs[layer] = []    
                gan_activation = gan.retained_layer(layer, clear = True)
                gan_activs[layer].append(gan_activation)



            #### Prepare images for CLIP
            img = (img+1)/2
            img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
            img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)

            #### Forward through CLIP
            _ = clipmodel.model.encode_image(img)

            del img

            clip_activs = {}
            for layer in clip_layers:
                clip_activs[layer] = []
                clip_activation = clipmodel.retained_layer(layer, clear = True)
                clip_activs[layer].append(clip_activation)



            batch_gan_stats_list = []
            for layer in gan_layers:
                gan_activs[layer] = torch.cat(gan_activs[layer], 0) #images x channels x m x m
                gan_activs[layer] = torch.permute(gan_activs[layer], (1,0,2,3)).contiguous() #channels x images x m x m
                gan_activs[layer] = gan_activs[layer].view(gan_activs[layer].shape[0], -1) 
                batch_gan_stats_list.append([torch.mean(gan_activs[layer],dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                      torch.std(gan_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])

            del gan_activs
            gan_stats_list.append(batch_gan_stats_list)


            batch_clip_stats_list = []
            clip_stats_list.append(batch_clip_stats_list)
            for layer in clip_layers:
                clip_activs[layer] = torch.cat(clip_activs[layer], 0)
                clip_activs[layer] = torch.permute(clip_activs[layer], (1,0,2,3)).contiguous()
                clip_activs[layer] = clip_activs[layer].view(clip_activs[layer].shape[0], -1)
                batch_clip_stats_list.append([torch.mean(clip_activs[layer], dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                      torch.std(clip_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])



            del clip_activs
            torch.cuda.empty_cache()

    
    ####################### After iterating
        print("Done Iteration for Stats")
        final_clip_stats_list = []

        for iii in range(len(batch_clip_stats_list)):
            means = torch.zeros_like(batch_clip_stats_list[iii][0])
            stds = torch.zeros_like(batch_clip_stats_list[iii][1])
            for jjj in range(epochs):
                means+=clip_stats_list[jjj][iii][0]
                stds+=clip_stats_list[jjj][iii][1]**2

            final_clip_stats_list.append([means/epochs, torch.sqrt(stds/epochs)])



        final_gan_stats_list = []

        for iii in range(len(batch_gan_stats_list)):
            means = torch.zeros_like(batch_gan_stats_list[iii][0])
            stds = torch.zeros_like(batch_gan_stats_list[iii][1])
            for jjj in range(epochs):
                means+=gan_stats_list[jjj][iii][0]
                stds+=gan_stats_list[jjj][iii][1]**2

            final_gan_stats_list.append([means/epochs, torch.sqrt(stds/epochs)])




    return final_gan_stats_list, final_clip_stats_list