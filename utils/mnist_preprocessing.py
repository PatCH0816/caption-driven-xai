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
  def __init__(self, root='./data', env='train', transform=None, target_transform=None, color=True, filter=range(10), color_split=5):
    super(DatasetMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    self.color = color
    self.env = env
    self._filter = filter
    self._color_split = color_split
    
    
    if self.color:
      self.prefix = 'color_'
    else:
      self.prefix = 'grey_'

    self.prepare_colored_mnist()
    if env in ['train', 'val', 'test', 'test_fool']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'mnist', f"{self.prefix}{env}.pt"))
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
        if ((idx % 5000-1) == 0):
            print(f'Converting {phase} image {idx+1}/{len(datasource)}')
        
      # http://yann.lecun.com/exdb/mnist/
      # try balanced emnist: https://www.nist.gov/itl/products-and-services/emnist-dataset
      print('Preparing Colored MNIST')
      # train_mnist = datasets.mnist.EMNIST(self.root, train=True, download=True, split="balanced") # 60'000 samples for training
      # test_mnist = datasets.mnist.EMNIST(self.root, train=False, download=True, split="balanced") # 10'000 samples for validation and test
      train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True, transform = transforms.Compose([transforms.ToTensor()])) # 60'000 samples for training
      test_mnist = datasets.mnist.MNIST(self.root, train=False, download=True, transform = transforms.Compose([transforms.ToTensor()])) # 10'000 samples for validation and test
      
      # shuffle datasets
      train_loader = torch.utils.data.DataLoader(train_mnist,
                                                batch_size=1,
                                                shuffle=True)
      test_loader = torch.utils.data.DataLoader(test_mnist,
                                                batch_size=1,
                                                shuffle=True)
      
      # setup
      environment = {"train":      {"dataset":train_loader},
                      "validation": {"dataset":train_loader},
                      "test":       {"dataset":test_loader},
                      "test_fool":  {"dataset":test_loader}}
      
      # split train/validation
      train_size = 50000

      train_set = []
      validation_set = []
      test_set = []
      test_fool_set = []
            
      for phase in environment.keys():        
        for idx, (im, ground_truth) in enumerate(environment[phase]["dataset"]):   
          # turn training data into train/validation datasets
          if (phase == "train" and idx >= train_size):
            continue
          if (phase == "validation" and idx < train_size):
            continue
          
          # apply filter
          if ground_truth not in self._filter:
            continue
            
          # progress bar
          conversion_progress(idx, environment[phase]["dataset"], phase)
                
          # Assign binary digit label for small=0 and large=1 numbers
          low_high_label = 1 if ground_truth > self._color_split else 0
        
          if self.color:
            # Assign random color labels to test set
            if phase == 'train':
              color_label = low_high_label
            elif phase == 'validation':
              color_label = low_high_label
            elif phase == 'test':
              color_label = low_high_label
            elif phase == 'test_fool':
              color_label = low_high_label^1 #np.random.choice([0,1])
            else:
              raise Exception("Oops.. Unkown phase!")
            
            # Color the digit
            new_image = color_grayscale_arr(np.array(im.squeeze()), green=color_label)
          else:
            # 3d grayscale mnist
            new_image = grayscale_3d_arr(np.array(im.squeeze()))
            color_label = 0
                               
          # create dataset with:
          # image (tensor format)
          # ground truth label (displayed number)
          # low_high_label (0 for low numbers, 1 for high numbers)
          # color_label (potentially mixed up label for test dataset)
          if phase == 'train':
            train_set.append((Image.fromarray((new_image * 255).astype(np.uint8)), ground_truth, low_high_label, color_label))
          elif phase == 'validation':
            validation_set.append((Image.fromarray((new_image * 255).astype(np.uint8)), ground_truth, low_high_label, color_label))
          elif phase == 'test':
            test_set.append((Image.fromarray((new_image * 255).astype(np.uint8)), ground_truth, low_high_label, color_label))
          elif phase == 'test_fool':
            test_fool_set.append((Image.fromarray((new_image * 255).astype(np.uint8)), ground_truth, low_high_label, color_label))
            
      print(len(train_set))
      print(len(validation_set))
      print(len(test_set))
      print(len(test_fool_set))
              
      return train_set, validation_set, test_set, test_fool_set
      
    # do files already exist?
    colored_mnist_dir = os.path.join(self.root, 'mnist')
    if os.path.exists(os.path.join(colored_mnist_dir, f"{self.prefix}train.pt")) \
      and os.path.exists(os.path.join(colored_mnist_dir, f"{self.prefix}val.pt")) \
      and os.path.exists(os.path.join(colored_mnist_dir, f"{self.prefix}test.pt")) \
      and os.path.exists(os.path.join(colored_mnist_dir, f"{self.prefix}test_fool.pt")):
      print('MNIST dataset already exists')
      return
    
    train_set, val_set, test_set, test_fool_set = mnist_grayscale_to_color()

    os.makedirs(colored_mnist_dir, exist_ok=True)
    torch.save(train_set, os.path.join(colored_mnist_dir, f"{self.prefix}train.pt"))
    torch.save(val_set, os.path.join(colored_mnist_dir, f"{self.prefix}val.pt"))
    torch.save(test_set, os.path.join(colored_mnist_dir, f"{self.prefix}test.pt"))
    torch.save(test_fool_set, os.path.join(colored_mnist_dir, f"{self.prefix}test_fool.pt"))