import clip
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

def clip_inference(model, preprocessed_images, texts, probabilities=False):
    """
    Returns either cosine similarity or probabilities
    """
    # preprocess images and texts
    image_input = torch.tensor(np.stack(preprocessed_images)).cuda()
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
    
    # invoke clip model
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
        
    # compute cosine similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    res = text_features @ image_features.T # similarity
    if probabilities:
        res = (100.0 * res).softmax(dim=-1)
    return res

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


def show_cosine_similarities(similarity, original_images, texts):
  nr_of_images = len(original_images)
  nr_of_texts = len(texts)
  
  plt.figure(figsize=(20, 14))
  plt.imshow(similarity, vmin=0.1, vmax=0.3)
  # plt.colorbar()
  plt.yticks(range(nr_of_texts), texts, fontsize=18)
  plt.xticks([])
  for i, image in enumerate(original_images):
      plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
  for x in range(similarity.shape[1]):
      for y in range(similarity.shape[0]):
          plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

  for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

  plt.xlim([-0.5, nr_of_images - 0.5])
  plt.ylim([nr_of_texts + 0.5, -2])

  plt.title("Cosine similarity between text and image features", size=20)
  

def show_text_img_probs(original_images, top_probs, top_labels, texts):
    plt.figure(figsize=(16, 16))

    for i, image in enumerate(original_images):
        plt.subplot(5, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(5, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [texts[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    plt.show()