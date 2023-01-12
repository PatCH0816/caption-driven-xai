import matplotlib.pyplot as plt
import numpy as np
import torch



def plot_history(hist, show_plots=['loss', 'acc'], show_points=False,
                  show_curves=['train_w_backprop', 'train', 'validation', 'test', 'test_fool'],
                  alternative_labels=[]):
    """
    Displays loss and accuracies from crossvalidation results.
    """
    # sanity check
    if len(alternative_labels) != 0 and len(alternative_labels) != len(show_curves):
        raise Exception("Every curve needs an alternative name! Mismatch in the length of lists!")
    
    if 'acc' in show_plots:
        
        # plot accuracies
        plt.subplot(1,2,1)
        
        for i, key in enumerate(show_curves):
            if show_points:
                # individual crossvalidation accuracies
                plt.plot(hist[key]['fold0']['acc'], 'x')
                plt.plot(hist[key]['fold1']['acc'], 'x')
                plt.plot(hist[key]['fold2']['acc'], 'x')
                plt.plot(hist[key]['fold3']['acc'], 'x')
                plt.plot(hist[key]['fold4']['acc'], 'x')

            # average accuracy
            avg_acc = np.vstack((hist[key]['fold0']['acc'],
                                    hist[key]['fold1']['acc'],
                                    hist[key]['fold2']['acc'],
                                    hist[key]['fold3']['acc'],
                                    hist[key]['fold4']['acc'])).mean(axis=0)
            
            if alternative_labels:
                plt.plot(avg_acc, label=alternative_labels[i])
            else:
                plt.plot(avg_acc, label=key)
            plt.grid()
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy in %")
            plt.legend()
                    
    if 'loss' in show_plots:
        
        # plot losses
        plt.subplot(1,2,2)
        
        for i, key in enumerate(show_curves):
            if show_points:
                # individual crossvalidation accuracies
                plt.plot(hist[key]['fold0']['loss'], 'x')
                plt.plot(hist[key]['fold1']['loss'], 'x')
                plt.plot(hist[key]['fold2']['loss'], 'x')
                plt.plot(hist[key]['fold3']['loss'], 'x')
                plt.plot(hist[key]['fold4']['loss'], 'x')

            # average accuracy
            avg_loss = np.vstack((hist[key]['fold0']['loss'],
                                    hist[key]['fold1']['loss'],
                                    hist[key]['fold2']['loss'],
                                    hist[key]['fold3']['loss'],
                                    hist[key]['fold4']['loss'])).mean(axis=0)
            
            if alternative_labels:
                plt.plot(avg_loss, label=alternative_labels[i])
            else:
                plt.plot(avg_loss, label=key)
            plt.grid()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

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