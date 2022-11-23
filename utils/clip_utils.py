import clip
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

#TODO merge assess_performance, clip_inference and harmonize show_examples_0_to_9

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


def text_feature_generator(clip_version, model, classnames, class_template):
    """
    Generates the text-feature matrix from given template sentences and classes and place it on the GPU.
    """
    with torch.no_grad():
        text_features = []
        for classname in classnames:
            texts = [class_template.format(template) for template in classname] # generate texts using templates with classes
            texts = clip_version.tokenize(texts).cuda() # generate text-tokens
            class_embeddings = model.encode_text(texts) # generate text embeddings -> torch.Size([nr_templates x 1024])
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # normalize feature vector -> torch.Size([nr_templates x 1024])
            class_embedding = class_embeddings.mean(dim=0) # average over all template sentences
            class_embedding /= class_embedding.norm() # normalize feature vector -> torch.Size([1024])
            text_features.append(class_embedding) # generate feature matrix -> torch.Size([nr_classes x 1024])
        text_features = torch.stack(text_features, dim=1).cuda()
    return text_features


def assess_performance(clip_version, model, preprocess, class_labels, class_template, dataset_loader, dataset_name="Dataset name"):
    transform = transforms.ToPILImage()
    
    # building text features
    text_features = text_feature_generator(clip_version, model, class_labels, class_template)
    
    with torch.no_grad():
        top1, top3, n = 0., 0., 0.
        for images, ground_truth_label, _, _ in dataset_loader:
            
            # preprocess images
            images_new = []
            for img in images:
                images_new.append(preprocess(transform(img)))

            # building image features
            images = torch.tensor(np.stack(images_new)).cuda()
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features

            # measure accuracy
            acc1, acc3 = clip_accuracy(similarities, ground_truth_label.cuda(), topk=(1, 3))
            top1 += acc1
            top3 += acc3
            n += images.size(0)

    top1 = (top1 / n) * 100
    top3 = (top3 / n) * 100 

    print(f"{dataset_name} -> Top-1 accuracy: {top1:.2f}")
    print(f"{dataset_name} -> Top-3 accuracy: {top3:.2f}")


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
    

def text_feature_generator(clip_version, model, classnames, class_template):
    """
    Generates the text-feature matrix from given template sentences and classes and place it on the GPU.
    """
    with torch.no_grad():
        text_features = []
        for classname in classnames:
            texts = [class_template.format(template) for template in classname] # generate texts using templates with classes
            texts = clip_version.tokenize(texts).cuda() # generate text-tokens
            class_embeddings = model.encode_text(texts) # generate text embeddings -> torch.Size([nr_templates x 1024])
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # normalize feature vector -> torch.Size([nr_templates x 1024])
            class_embedding = class_embeddings.mean(dim=0) # average over all template sentences
            class_embedding /= class_embedding.norm() # normalize feature vector -> torch.Size([1024])
            text_features.append(class_embedding) # generate feature matrix -> torch.Size([nr_classes x 1024])
        text_features = torch.stack(text_features, dim=1).cuda()
    return text_features
    
    
def clip_similarities(clip_version, model, preprocess, test_loader_color, descriptions, dataset_name="Dataset name"):
    return 42
    
def clip_accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy of a classifier given the predictions (logits) and the ground-truth labels (target).
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def clip_mnist_similarity(clip_version, model, preprocess, class_labels, class_template, dataset_loader, dataset_name="Dataset name"):
    transform = transforms.ToPILImage()
    
    # building text features
    text_features = text_feature_generator(clip_version, model, class_labels, class_template)
    
    with torch.no_grad():
        for images, ground_truth_label, low_high_label, color_label in dataset_loader:
            
            # preprocess images
            images_new = []
            for img in images:
                images_new.append(preprocess(transform(img)))

            # building image features
            images = torch.tensor(np.stack(images_new)).cuda()
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features

    return similarities.cpu(), low_high_label.cpu()


def clip_mnist_binary_accuracy(similarities, true_labels):
    return np.round(100.0 * (true_labels == similarities.argmax(axis=1)).sum().item() / len(true_labels), 2)
