import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt



def find_0_to_9_samples(dataset):
    """
    Returns random 10 indexes for the numbers 0-9 from a provided dataset
    """
    nroi_idx = {} # indexes of number of interest

    for i in range(10):
        while True:
            idx = np.random.randint(len(dataset))
            if dataset[idx][1] not in nroi_idx:
                nroi_idx[dataset[idx][1]] = idx
                break

    return nroi_idx



def get_model_paths(filename):
    """
    Get filepaths to backup model, history and logfile.
    """
    model_path = "/mnt/data/patrick.koller/masterthesis/data/results" + filename + ".mdl"
    history_path = "/mnt/data/patrick.koller/masterthesis/data/results" + filename + ".hist"
    log_path = "/mnt/data/patrick.koller/masterthesis/data/results" + filename + ".log"
    return model_path, history_path, log_path



def show_examples_0_to_9(dataset, preprocess, descriptions):
    original_images = []
    images = []
    texts = []
    
    transform = transforms.ToPILImage()
    nroi_idx = find_0_to_9_samples(dataset) # indexes for the numbers of interest (0-9)

    plt.figure(figsize=(16, 5))
    
    for idx in range(10):
        
        image = transform(dataset[nroi_idx[idx]][0])
        ground_truth_label = dataset[nroi_idx[idx]][1]
    
        plt.subplot(2, 5, len(images) + 1)
        plt.imshow(image)
        plt.title(f"{descriptions[str(ground_truth_label)]}")
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[str(ground_truth_label)])

    plt.tight_layout()
    return original_images, images, texts