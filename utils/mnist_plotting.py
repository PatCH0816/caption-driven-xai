import matplotlib.pyplot as plt
import numpy as np
import torch



def plot_history(hist, show_plots=['loss', 'acc'], show_curves=['train_w_backprop', 'train', 'validation', 'test', 'test_fool']):
    """
    Plot the losses and accuracies during the training, validation and test procedures.
    """
    if 'loss' in show_plots:
        plt.subplot(1,2,1)
        if 'train_w_backprop' in show_curves:
            plt.semilogy(range(len(hist['train_w_backprop']['loss'])), hist['train_w_backprop']['loss'], label='Train batch accumulated')
        if 'train' in show_curves:
            plt.semilogy(range(len(hist['train']['loss'])), hist['train']['loss'], label='Train')
        if 'validation' in show_curves:
            plt.semilogy(range(len(hist['validation']['loss'])), hist['validation']['loss'], label='Validation')
        if 'test' in show_curves:
            plt.semilogy(range(len(hist['test']['loss'])), hist['test']['loss'], label='Test')
        if 'test_fool' in show_curves:
            plt.semilogy(range(len(hist['test_fool']['loss'])), hist['test_fool']['loss'], label='Test fool')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

    if 'acc' in show_plots:
        plt.subplot(1,2,2)
        if 'train_w_backprop' in show_curves:
            plt.plot(range(len(hist['train_w_backprop']['acc'])), hist['train_w_backprop']['acc'], label='Train batch accumulated')
        if 'train' in show_curves:
            plt.plot(range(len(hist['train']['acc'])), hist['train']['acc'], label='Train')
        if 'validation' in show_curves:
            plt.plot(range(len(hist['validation']['acc'])), hist['validation']['acc'], label='Validation')
        if 'test' in show_curves:
            plt.plot(range(len(hist['test']['acc'])), hist['test']['acc'], label='Test')
        if 'test_fool' in show_curves:
            plt.plot(range(len(hist['test_fool']['acc'])), hist['test_fool']['acc'], label='Test fool')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy in %')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()



def digit_distribution(datasource):
    """
    Displays a bar plot showing the number of 5s and 8s in the dataset.
    """
    nr_of_5s = 0
    
    for i in range(len(datasource)):
        nr_of_5s += (datasource[i][1].item() == 5)
    
    plt.bar((5,8), (nr_of_5s, len(datasource) - nr_of_5s))
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Sample distribution")
    plt.show()
    


def binary_to_5_8(binary_pred):
    """
    Converts binary 0/1 to the digits 5/8
    """
    digit_set = (5,8)
    return digit_set[binary_pred]



def plot_digits(dataset, preprocessor):
    """
    Plots some digits from a provided colored MNIST dataset to be analyzed.
    Normalize(..) is expected to be the last transformation in the preprocessor.
    """
    fig = plt.figure(figsize=(13,8))
    columns = 6
    rows = 3
    ax = []
    
    normalizer = preprocessor.transforms.copy().pop()
    
    for i in range(columns*rows):
        img, _, true_label, color_label = dataset[i]
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("True label: " + str(binary_to_5_8(true_label)) + 
                         "\nColor label: " + str(binary_to_5_8(color_label)) +
                         "\nFlipped: " + str(true_label != color_label))
        
        img = torch.stack((img[0]*torch.Tensor(normalizer.std)[0] + torch.Tensor(normalizer.mean)[0],
                           img[1]*torch.Tensor(normalizer.std)[1] + torch.Tensor(normalizer.mean)[1],
                           img[2]*torch.Tensor(normalizer.std)[2] + torch.Tensor(normalizer.mean)[2]))
        plt.imshow(np.transpose(img.cpu().numpy(), (1,2,0)))
                        
    plt.tight_layout()
    plt.show()



def random_tests(dataset, model, device, preprocessor):
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
    normalizer = preprocessor.transforms.copy().pop()

    for i in range(columns*rows):
        ax.append(fig.add_subplot(8, 4, i + 1))
        if (pred[i].item() == true_label[i].item()):
            ax[-1].set_title("Ground truth: " + 
                             str(binary_to_5_8(true_label[i].item())) + 
                             "\nPrediction: " + 
                             str(binary_to_5_8(np.round(pred[i].item()))) + 
                             "\nCorrect!")
        else:
            ax[-1].set_title("Ground truth: " + 
                             str(binary_to_5_8(true_label[i].item())) + 
                             "\nPrediction: " + 
                             str(binary_to_5_8(np.round(pred[i].item()))) + 
                             "\nFooled!")
                        
        img[i] = torch.stack((img[i][0]*torch.Tensor(normalizer.std)[0] + torch.Tensor(normalizer.mean)[0],
                           img[i][1]*torch.Tensor(normalizer.std)[1] + torch.Tensor(normalizer.mean)[1],
                           img[i][2]*torch.Tensor(normalizer.std)[2] + torch.Tensor(normalizer.mean)[2]))
        plt.imshow(np.transpose(img[i].cpu().numpy(), (1,2,0)))
        
    plt.tight_layout()
    plt.show()