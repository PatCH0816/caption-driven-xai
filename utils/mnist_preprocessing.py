import os
from PIL import Image
import numpy as np
import torch
from torchvision import datasets


def color_grayscale_arr(arr, green=True):
  """
  Convert grayscale MNIST images to either red or green MNIST images by expanding
  the image to three RGB dimensions. The grayscale image either gets applied to
  the red or green channel.
  """
  assert arr.ndim == 2
  
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  
  if green:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)],axis=2)
  return arr


def grayscale_3d_arr(arr):
  """
  Convert 2d grayscale MNIST images to 3d grayscale MNIST images.
  """
  assert arr.ndim == 2
  
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])

  arr = np.concatenate([arr,
                        arr,
                        arr], axis=2)
  return arr


class DatasetMNIST(datasets.VisionDataset):
  """
  Downloads the grayscale MNIST dataset and transforms it into a colored MNIST dataset.
  Digits smaller than 5 are colored red for the train and validation set. Numbers larger
  than 5 are colored green for the train and validation set. The colors of the digits have
  a 50% probability to be flipped.
  """
  def __init__(self, root='./data', env='train', transform=None, target_transform=None, color=True):
    super(DatasetMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform,
                                color=color)
    self.color = color
    
    if self.color:
      self.prefix = 'color_'
    else:
      self.prefix = 'grey_'

    self.prepare_colored_mnist()
    if env in ['train', 'val', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'mnist', env) + '.pt')
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train, val and test')

  def __getitem__(self, index):
    """
    Overriden method from datasets.VisionDataset to apply transformations to the data
    before providing them to the dataloader.
    """
    img, ground_truth_label, low_high_label, color_label = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      low_high_label = self.target_transform(low_high_label)

    return img, ground_truth_label, low_high_label, color_label

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    """
    Download 60'000 grayscale MNIST images, split them into a train, validation and a test
    dataset and transform them to a colored MNIST dataset.
    """
    
    def mnist_grayscale_to_color():
      def conversion_progress(idx, datasource, phase='train'):
        if idx % 5000 == 0:
            print(f'Converting {phase} image {idx}/{len(datasource)}')
      
      # http://yann.lecun.com/exdb/mnist/
      print('Preparing Colored MNIST')
      train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True) # 60'000 samples for training
      test_mnist = datasets.mnist.MNIST(self.root, train=False, download=True) # 10'000 samples for validation and test

      train_set = []
      validation_set = []
      test_set = []
      
      for dataset in ['train_ds', 'test_ds']:
        datasource = train_mnist if (dataset == 'train_ds') else test_mnist
        
        for idx, (im, ground_truth) in enumerate(datasource):
          # determine train, validation, test phase/split
          if dataset == 'train_ds':
            if idx < 50000:
              phase = 'train'
            else:
              phase = 'validation'
          elif dataset == 'test_ds':
            phase = 'test'
          
          # progress bar
          conversion_progress(idx, datasource, phase)
                
          # Assign binary digit label for small and large numbers
          low_high_label = 1 if ground_truth > 4 else 0

          if self.color:
            # Assign random color labels to test set
            if phase == 'test':
              color_label = np.random.choice([0,1])
            else:
              color_label = low_high_label
            
            # Color the digit
            new_image = color_grayscale_arr(np.array(im), green=color_label)
          else:
            # 3d grayscale mnist
            new_image = grayscale_3d_arr(np.array(im))
            
          # create dataset with:
          # image (tensor format)
          # ground truth label (displayed number)
          # low_high_label (0 for low numbers, 1 for high numbers)
          # color_label (potentially mixed up label for test dataset)
          if phase == 'train':
            train_set.append((Image.fromarray(new_image), ground_truth, low_high_label, color_label))
          elif phase == 'validation':
            validation_set.append((Image.fromarray(new_image), ground_truth, low_high_label, color_label))
          else:
            test_set.append((Image.fromarray(new_image), ground_truth, low_high_label, color_label))
            
      return train_set, validation_set, test_set
      
    
    colored_mnist_dir = os.path.join(self.root, 'mnist')
    if os.path.exists(os.path.join(colored_mnist_dir, f"{self.prefix}train.pt")) \
        and os.path.exists(os.path.join(colored_mnist_dir, f"{self.prefix}val.pt")) \
        and os.path.exists(os.path.join(colored_mnist_dir, f"{self.prefix}test.pt")):
      print('MNIST dataset already exists')
      return
    
    train_set, val_set, test_set = mnist_grayscale_to_color()

    os.makedirs(colored_mnist_dir, exist_ok=True)
    torch.save(train_set, os.path.join(colored_mnist_dir, f"{self.prefix}train.pt"))
    torch.save(val_set, os.path.join(colored_mnist_dir, f"{self.prefix}val.pt"))
    torch.save(test_set, os.path.join(colored_mnist_dir, f"{self.prefix}test.pt"))
