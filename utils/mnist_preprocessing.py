import os
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class align_emnist_like_mnist:
    """
    Align EMNIST images like MNIST images. For some dubious reason, the EMNIST images
    are flipped and rotated. (See its documentation)
    """
    def __init__(self):
        self.angle = 90

    def __call__(self, x):
        x = transforms.functional.hflip(x)
        return transforms.functional.rotate(img=x, angle=self.angle)
    

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
  Downloads the grayscale MNIST dataset and transforms it into a colored MNIST dataset with dynamic configurations.
  """
  def __init__(self, root='./data', env='train', opt_postfix="", transform=None, target_transform=None, color=True, filter=range(10),
               first_color_max_nr=5, preprocess=None, test_fool_random=False):
    super(DatasetMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    self.color = color
    self.env = env
    self._filter = filter
    self._first_color_max_nr = first_color_max_nr
    self._opt_postfix = opt_postfix
    self._preprocess = preprocess
    self._test_fool_random = test_fool_random
    
    self.prefix = 'color_' if self.color else 'grey_'

    self.prepare_colored_mnist()
    if env in ['train', 'test', 'test_fool']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'mnist', f"{env}_{self.prefix}{self._opt_postfix}.pt"))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train, test and test_fool.')

  def __getitem__(self, index):
    """
    Overriden method from datasets.VisionDataset to apply transformations to the data
    before providing them to the dataloader.
    """
    img, ground_truth_label, low_high_label, color_label = self.data_label_tuples[index]
    
    img = self.transform(img) # torch.Size([3, 28, 28])
    transform = transforms.ToPILImage()
    img = self._preprocess(transform(img)) # torch.Size([3, 224, 224])
    
    if self.target_transform is not None:
      low_high_label = self.target_transform(low_high_label)

    return img, ground_truth_label, low_high_label, color_label

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    """
    Download 60'000 grayscale MNIST images, split them into a train, test and test_fool
    datasets and transform them to a colored MNIST dataset.
    """
    
    def mnist_grayscale_to_color():
      # http://yann.lecun.com/exdb/mnist/
      # try balanced emnist: https://www.nist.gov/itl/products-and-services/emnist-dataset
      print('Preparing Colored MNIST')
      train_mnist = datasets.mnist.EMNIST(self.root,
                                          train=True,
                                          download=True,
                                          split="mnist",
                                          transform = transforms.Compose([transforms.ToTensor(),
                                                                          align_emnist_like_mnist()])) # 60'000 samples for training
      
      test_mnist = datasets.mnist.EMNIST(self.root,
                                         train=False,
                                         download=True,
                                         split="mnist",
                                         transform = transforms.Compose([transforms.ToTensor(),
                                                                         align_emnist_like_mnist()])) # 10'000 samples for test
      
      # train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True, transform = transforms.Compose([transforms.ToTensor()])) # 60'000 samples for training
      # test_mnist = datasets.mnist.MNIST(self.root, train=False, download=True, transform = transforms.Compose([transforms.ToTensor()])) # 10'000 samples for test
      
      # split training dataset in smaller training dataset and test-fool/real-world dataset
      train_test_fool = train_test_split(train_mnist, train_size=50000, stratify=train_mnist.train_labels)
      
      # shuffle datasets
      train_loader = torch.utils.data.DataLoader(train_test_fool[0],
                                                  batch_size=1,
                                                  shuffle=True)
      
      test_fool_loader = torch.utils.data.DataLoader(train_test_fool[1],
                                                      batch_size=1,
                                                      shuffle=True)
      
      test_loader = torch.utils.data.DataLoader(test_mnist,
                                                batch_size=1,
                                                shuffle=True)
      
      # setup
      environment = {"train":       {"dataset":train_loader},
                      "test_fool":  {"dataset":test_fool_loader},
                      "test":       {"dataset":test_loader}}
      
      train_set = []
      test_set = []
      test_fool_set = []
            
      for phase in environment.keys():        
        for idx, (im, ground_truth) in enumerate(environment[phase]["dataset"]):   

          # progress bar
          if ((idx % 5000) == 0):
              print(f'Scanning {phase} image {idx}/{len(environment[phase]["dataset"])}')
                
          # apply filter
          if ground_truth not in self._filter:
            continue
                    
          # Assign binary digit label for small=0 and large=1 numbers
          low_high_label = 1 if ground_truth > self._first_color_max_nr else 0
        
          if self.color:
            # Assign random color labels to test set
            if phase == 'train':
              color_label = low_high_label
            elif phase == 'test':
              color_label = low_high_label
            elif phase == 'test_fool':
              if self._test_fool_random:
                color_label = np.random.choice([0,1])
              else:
                color_label = low_high_label^1
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
          elif phase == 'test':
            test_set.append((Image.fromarray((new_image * 255).astype(np.uint8)), ground_truth, low_high_label, color_label))
          elif phase == 'test_fool':
            test_fool_set.append((Image.fromarray((new_image * 255).astype(np.uint8)), ground_truth, low_high_label, color_label))
               
      return train_set, test_set, test_fool_set
      
    # do files already exist?
    colored_mnist_dir = os.path.join(self.root, 'mnist')
    if os.path.exists(os.path.join(colored_mnist_dir, f"train_{self.prefix}{self._opt_postfix}.pt")) \
      and os.path.exists(os.path.join(colored_mnist_dir, f"test_{self.prefix}{self._opt_postfix}.pt")) \
      and os.path.exists(os.path.join(colored_mnist_dir, f"test_fool_{self.prefix}{self._opt_postfix}.pt")):
      print('MNIST dataset already exists')
      return
    
    train_set, test_set, test_fool_set = mnist_grayscale_to_color()

    os.makedirs(colored_mnist_dir, exist_ok=True)
    torch.save(train_set, os.path.join(colored_mnist_dir, f"train_{self.prefix}{self._opt_postfix}.pt"))
    torch.save(test_set, os.path.join(colored_mnist_dir, f"test_{self.prefix}{self._opt_postfix}.pt"))
    torch.save(test_fool_set, os.path.join(colored_mnist_dir, f"test_fool_{self.prefix}{self._opt_postfix}.pt"))
