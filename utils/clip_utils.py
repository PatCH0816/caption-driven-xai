import clip
import torch
import numpy as np
from torchvision import transforms

def asses_clip_performance(model, preprocess, data_loader, text_descriptions, dataset_name="Dataset"):
    # build text tokens
    text_tokens = clip.tokenize(text_descriptions).cuda()

    # variables
    running_corrects = 0
    nr_of_images = 0

    # transformation
    transform = transforms.ToPILImage()

    # preprocess images in batches
    for img_batch, ground_truth_label, _, _ in data_loader:
        images_new = []
        for img in img_batch:
            # process a batch of images
            images_new.append(preprocess(transform(img)))
            nr_of_images += 1
        
        # building image features
        image_input = torch.tensor(np.stack(images_new)).cuda()

        # inference
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()
            
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        running_corrects += torch.sum(text_probs.argmax(axis=1) == ground_truth_label.cuda()).item()

    print(f"{dataset_name} accuracy: {100.0 * running_corrects / nr_of_images}%")
