import matplotlib.pyplot as plt
import numpy as np
import torch



def plot_digits(dataset):
    """
    Plots some digits from a provided colored MNIST dataset to be analyzed.
    """
    fig = plt.figure(figsize=(13,8))
    columns = 6
    rows = 3
    ax = []
    
    for i in range(columns*rows):
        img, _, true_label, color_label = dataset[i]
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("True label: " + str(true_label) + 
                         "\nColor label: " + str(color_label) +
                         "\nFlipped: " + str(true_label != color_label))
        plt.imshow(np.transpose(img.cpu().numpy(), (1,2,0)))
        
    plt.tight_layout()
    plt.show()



def random_tests(dataset, model, device):
    """
    Test and plot some digits from a provided colored MNIST dataset to be analyzed.
    """
    fig = plt.figure(figsize=(25,25))
    columns = 4
    rows = 8
    ax = []
    
    img, _, true_label, color_label = next(iter(dataset))
    model.eval()
    logits = model(img.to(device))
    pred = logits.argmax(dim=1)

    print("Batch accuracy: " + str(100 * torch.sum(pred == true_label.to(device)).item() / true_label.shape[0]) + "%")

    for i in range(columns*rows):
        ax.append(fig.add_subplot(8, 4, i + 1))
        if (pred[i].item() == true_label[i].item()):
            ax[-1].set_title("Ground truth: " + 
                             str(true_label[i].item()) + 
                             "\nPrediction: " + 
                             str(np.round(pred[i].item())) + 
                             "\nCorrect!")
        else:
            ax[-1].set_title("Ground truth: " + 
                             str(true_label[i].item()) + 
                             "\nPrediction: " + 
                             str(np.round(pred[i].item())) + 
                             "\nFooled!")
        plt.imshow(np.transpose(img[i].cpu().numpy(), (1,2,0)))
        
    plt.tight_layout()
    plt.show()