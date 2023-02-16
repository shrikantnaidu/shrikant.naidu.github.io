---
layout: post
title: Face Generation with GAN
date: 2020-06-14T 20:46:10 +03:00
description: "This is meta description"
image: "assets/images/masonary-post/face-gen.jpg"
categories: 
  - "Deep Learning"
---

# Face Generation with GAN

In this project, you'll define and train a DCGAN on a dataset of faces. Your goal is to get a generator network to generate *new* images of faces that look as realistic as possible!

The project will be broken down into a series of tasks from **loading in data to defining and training adversarial networks**. At the end of the notebook, you'll be able to visualize the results of your trained Generator to see how it performs; your generated samples should look like fairly realistic faces with small amounts of noise.

### Get the Data

You'll be using the [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train your adversarial networks.

This dataset is more complex than the number datasets (like MNIST or SVHN) you've been working with, and so, you should prepare to define deeper networks and train them for a longer time to get good results. It is suggested that you utilize a GPU for training.

### Pre-processed Data

Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. Some sample data is show below.

<!-- <img src="https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets//processed_face_data.png" width=60% /> -->

![png](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/output_9_0.png)

> If you are working locally, you can download this data [by clicking here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)

This is a zip file that you'll need to extract in the home directory of this notebook for further loading and processing. After extracting the data, you should be left with a directory of data `processed_celeba_small/`


```python
# can comment out after executing
# !unzip processed_celeba_small.zip
```


```python
data_dir = 'processed_celeba_small/'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import problem_unittests as tests
#import helper

%matplotlib inline
```

## Visualize the CelebA Data

The [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations. Since you're going to be generating faces, you won't need the annotations, you'll only need the images. Note that these are color images with [3 color channels (RGB)](https://en.wikipedia.org/wiki/Channel_(digital_image)#RGB_Images) each.

### Pre-process and Load the Data

Since the project's main focus is on building the GANs, we've done *some* of the pre-processing for you. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images. This *pre-processed* dataset is a smaller subset of the very large CelebA data.

> There are a few other steps that you'll need to **transform** this data and create a **DataLoader**.

#### Exercise: Complete the following `get_dataloader` function, such that it satisfies these requirements:

* Your images should be square, Tensor images of size `image_size x image_size` in the x and y dimension.
* Your function should return a DataLoader that shuffles and batches these Tensor images.

#### ImageFolder

To create a dataset given a directory of images, it's recommended that you use PyTorch's [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) wrapper, with a root directory `processed_celeba_small/` and data transformation passed in.


```python
# necessary imports
import torch
from torchvision import datasets
from torchvision import transforms
```


```python
def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """
    
    # TODO: Implement function and return a dataloader
    # resize and normalize the images
    transform = transforms.Compose([transforms.Resize(image_size), 
                                    transforms.ToTensor()])

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(data_dir, transform)

    # create and return DataLoaders
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader
```

## Create a DataLoader

#### Exercise: Create a DataLoader `celeba_train_loader` with appropriate hyperparameters.

Call the above function and create a dataloader to view images. 
* You can decide on any reasonable `batch_size` parameter
* Your `image_size` **must be** `32`. Resizing the data to a smaller size will make for faster training, while still creating convincing images of faces!


```python
# Define function hyperparameters
batch_size = 32
img_size = 32

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)
```

Next, you can view some images! You should seen square images of somewhat-centered faces.

Note: You'll need to convert the Tensor images into a NumPy type and transpose the dimensions to correctly display an image, suggested `imshow` code is below, but it may not be perfect.


```python
# helper display function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# obtain one batch of training images
dataiter = iter(celeba_train_loader)
images, _ = dataiter.next() # _ for no labels

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(20, 4))
plot_size=20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
```


![png](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/output_9_0.png)
    


#### Exercise: Pre-process your image data and scale it to a pixel range of -1 to 1

You need to do a bit of pre-processing; you know that the output of a `tanh` activated generator will contain pixel values in a range from -1 to 1, and so, we need to rescale our training images to a range of -1 to 1. (Right now, they are in a range from 0-1.)


```python
# TODO: Complete the scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min,max = feature_range
    x = x*(max - min) + min
    return x
```


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# check scaled range
# should be close to -1 to 1
img = images[0]
scaled_img = scale(img)

print('Min: ', scaled_img.min())
print('Max: ', scaled_img.max())
```

    Min:  tensor(-0.9922)
    Max:  tensor(1.)
    

---
# Define the Model

A GAN is comprised of two adversarial networks, a discriminator and a generator.

## Discriminator

Your first task will be to define the discriminator. This is a convolutional classifier like you've built before, only without any maxpooling layers. To deal with this complex data, it's suggested you use a deep network with **normalization**. You are also allowed to create any helper functions that may be useful.

#### Exercise: Complete the Discriminator class
* The inputs to the discriminator are 32x32x3 tensor images
* The output should be a single value that will indicate whether a given image is real or fake



```python
import torch.nn as nn
import torch.nn.functional as F
```


```python
# helper to build a convolution layer
def conv(in_channels,out_channels,kernel_size,stride = 2,padding = 1,batch_norm = True):
    layers = []
    conv_layer = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,
                      kernel_size = kernel_size,stride = stride,padding = padding,bias= False)
    
    layers.append(conv_layer)
    if batch_norm == True:
        layers.append(nn.BatchNorm2d(out_channels))
    
    return nn.Sequential(*layers)
```


```python
class Discriminator(nn.Module):

    def __init__(self, conv_dim=32):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim
        # covolution layers
        
        # input 32 x 32 x 3 -> output 16 x 16 x 32
        self.conv1 = conv(3,conv_dim,4,batch_norm = False)
        # input 16 x 16 x 32 ->  output 8 x 8 x 64
        self.conv2 = conv(conv_dim,conv_dim*2,4)
        # input 8 x 8 x 64 -> output 4 x 4 x 128
        self.conv3 = conv(conv_dim*2,conv_dim*4,4)
        # input 4 x 4 x 128 -> output 2 x 2 x 256
        self.conv4 = conv(conv_dim*4,conv_dim*8,4)
        
        # classification layers
        self.fc = nn.Linear(conv_dim*8*2*2,1)
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = F.leaky_relu(self.conv3(x),0.2)
        x = F.leaky_relu(self.conv4(x),0.2)
        
        # output
        x = x.view(-1,self.conv_dim*8*2*2)
        x = self.fc(x)
    
        return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_discriminator(Discriminator)
```

    Tests Passed
    

## Generator

The generator should upsample an input and generate a *new* image of the same size as our training data `32x32x3`. This should be mostly transpose convolutional layers with normalization applied to the outputs.

#### Exercise: Complete the Generator class
* The inputs to the generator are vectors of some length `z_size`
* The output should be a image of shape `32x32x3`


```python
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    # create a sequence of transpose + optional batch norm layers
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                                              kernel_size, stride, padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)
    
    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)
```


```python
class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim = 32):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        self.conv_dim = conv_dim
        
        self.fc = nn.Linear(z_size,conv_dim*8*2*2)
        
        self.t_conv1 = deconv(conv_dim*8,conv_dim*4,4)
        self.t_conv2 = deconv(conv_dim*4,conv_dim*2,4)
        self.t_conv3 = deconv(conv_dim*2,conv_dim,4)
        self.t_conv4 = deconv(conv_dim,3,4,batch_norm = False)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = x.view(-1,self.conv_dim*8,2,2)
        
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.tanh(self.t_conv4(x))
        
        return x

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_generator(Generator)
```

    Tests Passed
    

## Initialize the weights of your networks

To help your models converge, you should initialize the weights of the convolutional and linear layers in your model. From reading the [original DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf), they say:
> All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.

So, your next task will be to define a weight initialization function that does just this!

You can refer back to the lesson on weight initialization or even consult existing model code, such as that from [the `networks.py` file in CycleGAN Github repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py) to help you complete this function.

#### Exercise: Complete the weight initialization function

* This should initialize only **convolutional** and **linear** layers
* Initialize the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
* The bias terms, if they exist, may be left alone or set to 0.


```python
def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__
    
    # TODO: Apply initial weights to convolutional and linear layers
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.fill_(0)
```

## Build complete network

Define your models' hyperparameters and instantiate the discriminator and generator from the classes defined above. Make sure you've passed in the correct input arguments.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G
```

#### Exercise: Define model hyperparameters


```python
# Define model hyperparams
d_conv_dim = 32
g_conv_dim = 32
z_size = 100

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
D, G = build_network(d_conv_dim, g_conv_dim, z_size)
```

    Discriminator(
      (conv1): Sequential(
        (0): Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (conv2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (fc): Linear(in_features=1024, out_features=1, bias=True)
    )
    
    Generator(
      (fc): Linear(in_features=100, out_features=1024, bias=True)
      (t_conv1): Sequential(
        (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv2): Sequential(
        (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv3): Sequential(
        (0): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (t_conv4): Sequential(
        (0): ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
    )
    

### Training on GPU

Check if you can train on GPU. Here, we'll set this as a boolean variable `train_on_gpu`. Later, you'll be responsible for making sure that 
>* Models,
* Model inputs, and
* Loss function arguments

Are moved to GPU, where appropriate.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Training on GPU!')
```

    Training on GPU!
    

---
## Discriminator and Generator Losses

Now we need to calculate the losses for both types of adversarial networks.

### Discriminator Losses

> * For the discriminator, the total loss is the sum of the losses for real and fake images, `d_loss = d_real_loss + d_fake_loss`. 
* Remember that we want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.


### Generator Loss

The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to *think* its generated images are *real*.

#### Exercise: Complete real and fake loss functions

**You may choose to use either cross entropy or a least squares error loss to complete the following `real_loss` and `fake_loss` functions.**


```python
def real_loss(D_out,smooth=False):
    batch_size = D_out.size(0)
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    # move labels to GPU if available     
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss
```

## Optimizers

#### Exercise: Define optimizers for your Discriminator (D) and Generator (G)

Define optimizers for your models with appropriate hyperparameters.


```python
import torch.optim as optim

# Create optimizers for the discriminator D and generator G
lr = 0.0002
beta1=0.5
beta2=0.999 # default value

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])
```

---
## Training

Training will involve alternating between training the discriminator and the generator. You'll use your functions `real_loss` and `fake_loss` to help you calculate the discriminator losses.

* You should train the discriminator by alternating on real and fake images
* Then the generator, which tries to trick the discriminator and should have an opposing loss function


#### Saving Samples

You've been given some code to print out some loss statistics and save some generated "fake" samples.

#### Exercise: Complete the training function

Keep in mind that, if you've moved your models to GPU, you'll also have to move any model inputs to GPU.


```python
def train(D, G, n_epochs, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================
            
            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
        
            d_optimizer.zero_grad()
            # 1. Train the discriminator on real and fake images
            
            # Train with real images
            if train_on_gpu:
                real_images = real_images.cuda()
            
            D_real = D(real_images)
            d_real_loss = real_loss(D_real)
            
            # 2. Train with fake images
        
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            
            # Compute the discriminator losses on fake images            
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

             
            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # Compute the discriminator losses on fake images 
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake) # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step()                
                
            
            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs, d_loss.item(), g_loss.item()))


        ## AFTER EACH EPOCH##    
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval() # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train() # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    
    # finally return losses
    return losses
```

Set your number of training epochs and train your GAN!


```python
# set number of epochs 
n_epochs = 10


"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# call training function
losses = train(D, G, n_epochs=n_epochs)
```

    Epoch [    1/   10] | d_loss: 1.4375 | g_loss: 0.8283
    Epoch [    1/   10] | d_loss: 0.1367 | g_loss: 3.4099
    Epoch [    1/   10] | d_loss: 0.0330 | g_loss: 4.5209
    Epoch [    1/   10] | d_loss: 0.0989 | g_loss: 4.5074
    Epoch [    1/   10] | d_loss: 0.2173 | g_loss: 3.7985
    Epoch [    1/   10] | d_loss: 0.2060 | g_loss: 3.5710
    Epoch [    1/   10] | d_loss: 0.4686 | g_loss: 4.2082
    Epoch [    1/   10] | d_loss: 0.9417 | g_loss: 4.9502
    Epoch [    1/   10] | d_loss: 0.1812 | g_loss: 2.7069
    Epoch [    1/   10] | d_loss: 0.2640 | g_loss: 3.7460
    Epoch [    1/   10] | d_loss: 0.3547 | g_loss: 3.1814
    Epoch [    1/   10] | d_loss: 0.9886 | g_loss: 1.3821
    Epoch [    1/   10] | d_loss: 0.6250 | g_loss: 2.3265
    Epoch [    1/   10] | d_loss: 0.4463 | g_loss: 2.8969
    Epoch [    1/   10] | d_loss: 0.8534 | g_loss: 2.5419
    Epoch [    1/   10] | d_loss: 0.6156 | g_loss: 3.4758
    Epoch [    1/   10] | d_loss: 0.6725 | g_loss: 2.4061
    Epoch [    1/   10] | d_loss: 0.5060 | g_loss: 3.2003
    Epoch [    1/   10] | d_loss: 1.2175 | g_loss: 1.5792
    Epoch [    1/   10] | d_loss: 1.0904 | g_loss: 1.3760
    Epoch [    1/   10] | d_loss: 1.0209 | g_loss: 1.4324
    Epoch [    1/   10] | d_loss: 0.7405 | g_loss: 2.5089
    Epoch [    1/   10] | d_loss: 0.7077 | g_loss: 2.0425
    Epoch [    1/   10] | d_loss: 0.9027 | g_loss: 1.8268
    Epoch [    1/   10] | d_loss: 0.7777 | g_loss: 1.2721
    Epoch [    1/   10] | d_loss: 0.8589 | g_loss: 1.4374
    Epoch [    1/   10] | d_loss: 0.5048 | g_loss: 2.3149
    Epoch [    1/   10] | d_loss: 0.9330 | g_loss: 2.0192
    Epoch [    1/   10] | d_loss: 0.7803 | g_loss: 1.7245
    Epoch [    1/   10] | d_loss: 0.8362 | g_loss: 2.3229
    Epoch [    1/   10] | d_loss: 0.5941 | g_loss: 1.9812
    Epoch [    1/   10] | d_loss: 0.6449 | g_loss: 1.8468
    Epoch [    1/   10] | d_loss: 0.8615 | g_loss: 3.1905
    Epoch [    1/   10] | d_loss: 1.0127 | g_loss: 2.1786
    Epoch [    1/   10] | d_loss: 0.7481 | g_loss: 1.3780
    Epoch [    1/   10] | d_loss: 0.6624 | g_loss: 2.0557
    Epoch [    1/   10] | d_loss: 1.2710 | g_loss: 0.8236
    Epoch [    1/   10] | d_loss: 0.7089 | g_loss: 1.8478
    Epoch [    1/   10] | d_loss: 1.1152 | g_loss: 2.5029
    Epoch [    1/   10] | d_loss: 0.5635 | g_loss: 1.5920
    Epoch [    1/   10] | d_loss: 0.8962 | g_loss: 2.8575
    Epoch [    1/   10] | d_loss: 1.1699 | g_loss: 3.3993
    Epoch [    1/   10] | d_loss: 0.6756 | g_loss: 1.6613
    Epoch [    1/   10] | d_loss: 0.8714 | g_loss: 1.6527
    Epoch [    1/   10] | d_loss: 0.6948 | g_loss: 2.4757
    Epoch [    1/   10] | d_loss: 0.9373 | g_loss: 1.5293
    Epoch [    1/   10] | d_loss: 0.7582 | g_loss: 1.7391
    Epoch [    1/   10] | d_loss: 0.8760 | g_loss: 1.5461
    Epoch [    1/   10] | d_loss: 1.0955 | g_loss: 2.5232
    Epoch [    1/   10] | d_loss: 0.8702 | g_loss: 1.9970
    Epoch [    1/   10] | d_loss: 0.6099 | g_loss: 1.8245
    Epoch [    1/   10] | d_loss: 0.5941 | g_loss: 3.0635
    Epoch [    1/   10] | d_loss: 0.7388 | g_loss: 1.6954
    Epoch [    1/   10] | d_loss: 0.8215 | g_loss: 1.4895
    Epoch [    1/   10] | d_loss: 0.4918 | g_loss: 2.6645
    Epoch [    1/   10] | d_loss: 0.7824 | g_loss: 3.0270
    Epoch [    1/   10] | d_loss: 0.7255 | g_loss: 2.5206
    Epoch [    2/   10] | d_loss: 0.7361 | g_loss: 1.4261
    Epoch [    2/   10] | d_loss: 0.8221 | g_loss: 1.3983
    Epoch [    2/   10] | d_loss: 0.6529 | g_loss: 1.5358
    Epoch [    2/   10] | d_loss: 0.7648 | g_loss: 1.1755
    Epoch [    2/   10] | d_loss: 0.7850 | g_loss: 2.1602
    Epoch [    2/   10] | d_loss: 0.5967 | g_loss: 1.7183
    Epoch [    2/   10] | d_loss: 0.9293 | g_loss: 0.4311
    Epoch [    2/   10] | d_loss: 0.8048 | g_loss: 2.4040
    Epoch [    2/   10] | d_loss: 0.9546 | g_loss: 2.9014
    Epoch [    2/   10] | d_loss: 1.0130 | g_loss: 1.4458
    Epoch [    2/   10] | d_loss: 0.7787 | g_loss: 1.9187
    Epoch [    2/   10] | d_loss: 0.8206 | g_loss: 1.3583
    Epoch [    2/   10] | d_loss: 0.7602 | g_loss: 1.4859
    Epoch [    2/   10] | d_loss: 1.6796 | g_loss: 2.4626
    Epoch [    2/   10] | d_loss: 0.6030 | g_loss: 1.8397
    Epoch [    2/   10] | d_loss: 0.8352 | g_loss: 2.1232
    Epoch [    2/   10] | d_loss: 0.8560 | g_loss: 1.8031
    Epoch [    2/   10] | d_loss: 0.5210 | g_loss: 2.2766
    Epoch [    2/   10] | d_loss: 1.0136 | g_loss: 0.9756
    Epoch [    2/   10] | d_loss: 0.8236 | g_loss: 1.8405
    Epoch [    2/   10] | d_loss: 0.6088 | g_loss: 2.2920
    Epoch [    2/   10] | d_loss: 0.8901 | g_loss: 2.2165
    Epoch [    2/   10] | d_loss: 0.7736 | g_loss: 1.4659
    Epoch [    2/   10] | d_loss: 0.6071 | g_loss: 2.0560
    Epoch [    2/   10] | d_loss: 1.0470 | g_loss: 1.2345
    Epoch [    2/   10] | d_loss: 0.8429 | g_loss: 1.0077
    Epoch [    2/   10] | d_loss: 0.5577 | g_loss: 2.2328
    Epoch [    2/   10] | d_loss: 0.6432 | g_loss: 1.2822
    Epoch [    2/   10] | d_loss: 0.4028 | g_loss: 2.3859
    Epoch [    2/   10] | d_loss: 0.8212 | g_loss: 2.2920
    Epoch [    2/   10] | d_loss: 1.0978 | g_loss: 1.2951
    Epoch [    2/   10] | d_loss: 0.8459 | g_loss: 3.2531
    Epoch [    2/   10] | d_loss: 1.3283 | g_loss: 1.5369
    Epoch [    2/   10] | d_loss: 0.6483 | g_loss: 2.5114
    Epoch [    2/   10] | d_loss: 0.6488 | g_loss: 2.3417
    Epoch [    2/   10] | d_loss: 0.8230 | g_loss: 1.4507
    Epoch [    2/   10] | d_loss: 0.6827 | g_loss: 2.0984
    Epoch [    2/   10] | d_loss: 0.5202 | g_loss: 3.4786
    Epoch [    2/   10] | d_loss: 0.5568 | g_loss: 2.5354
    Epoch [    2/   10] | d_loss: 0.9407 | g_loss: 2.3356
    Epoch [    2/   10] | d_loss: 0.5611 | g_loss: 2.2820
    Epoch [    2/   10] | d_loss: 1.2914 | g_loss: 2.4045
    Epoch [    2/   10] | d_loss: 0.4088 | g_loss: 2.1361
    Epoch [    2/   10] | d_loss: 0.4458 | g_loss: 1.8634
    Epoch [    2/   10] | d_loss: 0.6059 | g_loss: 1.1045
    Epoch [    2/   10] | d_loss: 0.6917 | g_loss: 1.9551
    Epoch [    2/   10] | d_loss: 0.8114 | g_loss: 2.0670
    Epoch [    2/   10] | d_loss: 1.0485 | g_loss: 1.6118
    Epoch [    2/   10] | d_loss: 0.6789 | g_loss: 1.2525
    Epoch [    2/   10] | d_loss: 0.3771 | g_loss: 1.4703
    Epoch [    2/   10] | d_loss: 0.6499 | g_loss: 3.0064
    Epoch [    2/   10] | d_loss: 0.8662 | g_loss: 2.8859
    Epoch [    2/   10] | d_loss: 0.6973 | g_loss: 2.0897
    Epoch [    2/   10] | d_loss: 1.0380 | g_loss: 0.4704
    Epoch [    2/   10] | d_loss: 0.7165 | g_loss: 1.2300
    Epoch [    2/   10] | d_loss: 1.0388 | g_loss: 1.9755
    Epoch [    2/   10] | d_loss: 1.1867 | g_loss: 2.1088
    Epoch [    3/   10] | d_loss: 0.8097 | g_loss: 3.0393
    Epoch [    3/   10] | d_loss: 0.5019 | g_loss: 2.5709
    Epoch [    3/   10] | d_loss: 0.7984 | g_loss: 1.5304
    Epoch [    3/   10] | d_loss: 0.7146 | g_loss: 2.8520
    Epoch [    3/   10] | d_loss: 0.6904 | g_loss: 2.6993
    Epoch [    3/   10] | d_loss: 0.5089 | g_loss: 2.0663
    Epoch [    3/   10] | d_loss: 0.9721 | g_loss: 1.3061
    Epoch [    3/   10] | d_loss: 0.7447 | g_loss: 3.1119
    Epoch [    3/   10] | d_loss: 0.6252 | g_loss: 2.6550
    Epoch [    3/   10] | d_loss: 0.3589 | g_loss: 2.8221
    Epoch [    3/   10] | d_loss: 0.4794 | g_loss: 1.8572
    Epoch [    3/   10] | d_loss: 0.4721 | g_loss: 2.7487
    Epoch [    3/   10] | d_loss: 1.2881 | g_loss: 0.6078
    Epoch [    3/   10] | d_loss: 0.8306 | g_loss: 1.6300
    Epoch [    3/   10] | d_loss: 0.4302 | g_loss: 2.5741
    Epoch [    3/   10] | d_loss: 0.4856 | g_loss: 2.8508
    Epoch [    3/   10] | d_loss: 1.0638 | g_loss: 1.2981
    Epoch [    3/   10] | d_loss: 1.2009 | g_loss: 2.4391
    Epoch [    3/   10] | d_loss: 0.5120 | g_loss: 0.8365
    Epoch [    3/   10] | d_loss: 1.0198 | g_loss: 1.5580
    Epoch [    3/   10] | d_loss: 0.8820 | g_loss: 2.1524
    Epoch [    3/   10] | d_loss: 0.8794 | g_loss: 1.9807
    Epoch [    3/   10] | d_loss: 1.2404 | g_loss: 1.5583
    Epoch [    3/   10] | d_loss: 0.5065 | g_loss: 1.5696
    Epoch [    3/   10] | d_loss: 0.7311 | g_loss: 2.3812
    Epoch [    3/   10] | d_loss: 0.7586 | g_loss: 1.6321
    Epoch [    3/   10] | d_loss: 1.1888 | g_loss: 3.0969
    Epoch [    3/   10] | d_loss: 0.6019 | g_loss: 1.3327
    Epoch [    3/   10] | d_loss: 0.9333 | g_loss: 1.7940
    Epoch [    3/   10] | d_loss: 0.9408 | g_loss: 3.4469
    Epoch [    3/   10] | d_loss: 0.4765 | g_loss: 1.4752
    Epoch [    3/   10] | d_loss: 0.5436 | g_loss: 2.0790
    Epoch [    3/   10] | d_loss: 0.8359 | g_loss: 1.3764
    Epoch [    3/   10] | d_loss: 0.8532 | g_loss: 3.2155
    Epoch [    3/   10] | d_loss: 0.8930 | g_loss: 1.4274
    Epoch [    3/   10] | d_loss: 0.7218 | g_loss: 2.5226
    Epoch [    3/   10] | d_loss: 0.5686 | g_loss: 1.5422
    Epoch [    3/   10] | d_loss: 0.9219 | g_loss: 1.8003
    Epoch [    3/   10] | d_loss: 1.2073 | g_loss: 0.9530
    Epoch [    3/   10] | d_loss: 0.8309 | g_loss: 1.0924
    Epoch [    3/   10] | d_loss: 0.8689 | g_loss: 1.7498
    Epoch [    3/   10] | d_loss: 1.1529 | g_loss: 0.9297
    Epoch [    3/   10] | d_loss: 0.8227 | g_loss: 1.6970
    Epoch [    3/   10] | d_loss: 0.5959 | g_loss: 1.0709
    Epoch [    3/   10] | d_loss: 0.8132 | g_loss: 2.8963
    Epoch [    3/   10] | d_loss: 1.2516 | g_loss: 1.1911
    Epoch [    3/   10] | d_loss: 0.9279 | g_loss: 1.3907
    Epoch [    3/   10] | d_loss: 1.0880 | g_loss: 2.5361
    Epoch [    3/   10] | d_loss: 0.8747 | g_loss: 2.5947
    Epoch [    3/   10] | d_loss: 1.0380 | g_loss: 1.3927
    Epoch [    3/   10] | d_loss: 0.8980 | g_loss: 1.9233
    Epoch [    3/   10] | d_loss: 0.9106 | g_loss: 2.5329
    Epoch [    3/   10] | d_loss: 0.5308 | g_loss: 2.5881
    Epoch [    3/   10] | d_loss: 1.0285 | g_loss: 0.8125
    Epoch [    3/   10] | d_loss: 0.7343 | g_loss: 2.8655
    Epoch [    3/   10] | d_loss: 0.7057 | g_loss: 1.7360
    Epoch [    3/   10] | d_loss: 0.5431 | g_loss: 1.2880
    Epoch [    4/   10] | d_loss: 0.8401 | g_loss: 1.4240
    Epoch [    4/   10] | d_loss: 0.8616 | g_loss: 2.2292
    Epoch [    4/   10] | d_loss: 0.8513 | g_loss: 0.8143
    Epoch [    4/   10] | d_loss: 0.9198 | g_loss: 2.0126
    Epoch [    4/   10] | d_loss: 0.7864 | g_loss: 0.7880
    Epoch [    4/   10] | d_loss: 0.6304 | g_loss: 2.2339
    Epoch [    4/   10] | d_loss: 0.9616 | g_loss: 1.9981
    Epoch [    4/   10] | d_loss: 0.9861 | g_loss: 3.1358
    Epoch [    4/   10] | d_loss: 0.6280 | g_loss: 2.0134
    Epoch [    4/   10] | d_loss: 0.4167 | g_loss: 1.8948
    Epoch [    4/   10] | d_loss: 0.9732 | g_loss: 1.7885
    Epoch [    4/   10] | d_loss: 1.3507 | g_loss: 3.3076
    Epoch [    4/   10] | d_loss: 0.8582 | g_loss: 2.4259
    Epoch [    4/   10] | d_loss: 0.3222 | g_loss: 1.3931
    Epoch [    4/   10] | d_loss: 0.3784 | g_loss: 2.0218
    Epoch [    4/   10] | d_loss: 0.5218 | g_loss: 3.1135
    Epoch [    4/   10] | d_loss: 1.2346 | g_loss: 1.9277
    Epoch [    4/   10] | d_loss: 0.8060 | g_loss: 2.9532
    Epoch [    4/   10] | d_loss: 0.5906 | g_loss: 3.2836
    Epoch [    4/   10] | d_loss: 0.3541 | g_loss: 2.9068
    Epoch [    4/   10] | d_loss: 0.9875 | g_loss: 0.9372
    Epoch [    4/   10] | d_loss: 0.6712 | g_loss: 2.4969
    Epoch [    4/   10] | d_loss: 0.6518 | g_loss: 1.6130
    Epoch [    4/   10] | d_loss: 0.5608 | g_loss: 2.5040
    Epoch [    4/   10] | d_loss: 0.8226 | g_loss: 1.6709
    Epoch [    4/   10] | d_loss: 0.4274 | g_loss: 1.9891
    Epoch [    4/   10] | d_loss: 0.9803 | g_loss: 1.1438
    Epoch [    4/   10] | d_loss: 0.6038 | g_loss: 2.0693
    Epoch [    4/   10] | d_loss: 0.7240 | g_loss: 1.6234
    Epoch [    4/   10] | d_loss: 1.1355 | g_loss: 2.6563
    Epoch [    4/   10] | d_loss: 0.6979 | g_loss: 1.5845
    Epoch [    4/   10] | d_loss: 0.6271 | g_loss: 2.4673
    Epoch [    4/   10] | d_loss: 0.8741 | g_loss: 3.4031
    Epoch [    4/   10] | d_loss: 0.8035 | g_loss: 1.0717
    Epoch [    4/   10] | d_loss: 0.5466 | g_loss: 3.6925
    Epoch [    4/   10] | d_loss: 0.7398 | g_loss: 1.3586
    Epoch [    4/   10] | d_loss: 0.9093 | g_loss: 1.7668
    Epoch [    4/   10] | d_loss: 0.9914 | g_loss: 2.2887
    Epoch [    4/   10] | d_loss: 0.6766 | g_loss: 3.2470
    Epoch [    4/   10] | d_loss: 0.9100 | g_loss: 2.9327
    Epoch [    4/   10] | d_loss: 0.4537 | g_loss: 1.7861
    Epoch [    4/   10] | d_loss: 0.7286 | g_loss: 2.2573
    Epoch [    4/   10] | d_loss: 0.7621 | g_loss: 1.6164
    Epoch [    4/   10] | d_loss: 0.6815 | g_loss: 2.1096
    Epoch [    4/   10] | d_loss: 0.5489 | g_loss: 1.9791
    Epoch [    4/   10] | d_loss: 0.4194 | g_loss: 2.2744
    Epoch [    4/   10] | d_loss: 0.5542 | g_loss: 1.8974
    Epoch [    4/   10] | d_loss: 0.5394 | g_loss: 2.2294
    Epoch [    4/   10] | d_loss: 0.5985 | g_loss: 2.3418
    Epoch [    4/   10] | d_loss: 0.7873 | g_loss: 2.4874
    Epoch [    4/   10] | d_loss: 0.5031 | g_loss: 2.3360
    Epoch [    4/   10] | d_loss: 0.6192 | g_loss: 1.5156
    Epoch [    4/   10] | d_loss: 1.2282 | g_loss: 0.9613
    Epoch [    4/   10] | d_loss: 0.7149 | g_loss: 3.4017
    Epoch [    4/   10] | d_loss: 0.6998 | g_loss: 1.7441
    Epoch [    4/   10] | d_loss: 0.2832 | g_loss: 2.7605
    Epoch [    4/   10] | d_loss: 0.4165 | g_loss: 2.6288
    Epoch [    5/   10] | d_loss: 1.1319 | g_loss: 2.9711
    Epoch [    5/   10] | d_loss: 0.8193 | g_loss: 1.6927
    Epoch [    5/   10] | d_loss: 0.6316 | g_loss: 2.9231
    Epoch [    5/   10] | d_loss: 0.8828 | g_loss: 2.3579
    Epoch [    5/   10] | d_loss: 0.7815 | g_loss: 0.9519
    Epoch [    5/   10] | d_loss: 0.6774 | g_loss: 1.9494
    Epoch [    5/   10] | d_loss: 0.5645 | g_loss: 2.0426
    Epoch [    5/   10] | d_loss: 0.4826 | g_loss: 1.4202
    Epoch [    5/   10] | d_loss: 0.9609 | g_loss: 1.2853
    Epoch [    5/   10] | d_loss: 0.6564 | g_loss: 0.6472
    Epoch [    5/   10] | d_loss: 0.7649 | g_loss: 2.5583
    Epoch [    5/   10] | d_loss: 0.8553 | g_loss: 2.9331
    Epoch [    5/   10] | d_loss: 0.9975 | g_loss: 3.8976
    Epoch [    5/   10] | d_loss: 0.9121 | g_loss: 1.9868
    Epoch [    5/   10] | d_loss: 1.0603 | g_loss: 0.9347
    Epoch [    5/   10] | d_loss: 0.5517 | g_loss: 2.0222
    Epoch [    5/   10] | d_loss: 0.8386 | g_loss: 1.4041
    Epoch [    5/   10] | d_loss: 0.8956 | g_loss: 2.2966
    Epoch [    5/   10] | d_loss: 0.5073 | g_loss: 2.2514
    Epoch [    5/   10] | d_loss: 0.6612 | g_loss: 1.8508
    Epoch [    5/   10] | d_loss: 0.4714 | g_loss: 1.4664
    Epoch [    5/   10] | d_loss: 0.4838 | g_loss: 1.4921
    Epoch [    5/   10] | d_loss: 0.5782 | g_loss: 2.1109
    Epoch [    5/   10] | d_loss: 0.7387 | g_loss: 2.9436
    Epoch [    5/   10] | d_loss: 0.6819 | g_loss: 0.8043
    Epoch [    5/   10] | d_loss: 0.4835 | g_loss: 3.2234
    Epoch [    5/   10] | d_loss: 0.3402 | g_loss: 2.0674
    Epoch [    5/   10] | d_loss: 0.4534 | g_loss: 3.4563
    Epoch [    5/   10] | d_loss: 0.5540 | g_loss: 1.1556
    Epoch [    5/   10] | d_loss: 0.7765 | g_loss: 1.2208
    Epoch [    5/   10] | d_loss: 1.0301 | g_loss: 1.1580
    Epoch [    5/   10] | d_loss: 0.6595 | g_loss: 1.4798
    Epoch [    5/   10] | d_loss: 0.7811 | g_loss: 2.1072
    Epoch [    5/   10] | d_loss: 0.8951 | g_loss: 2.2031
    Epoch [    5/   10] | d_loss: 0.6046 | g_loss: 3.0415
    Epoch [    5/   10] | d_loss: 0.3505 | g_loss: 3.3982
    Epoch [    5/   10] | d_loss: 0.8029 | g_loss: 2.2610
    Epoch [    5/   10] | d_loss: 0.6194 | g_loss: 2.2170
    Epoch [    5/   10] | d_loss: 0.8113 | g_loss: 2.6177
    Epoch [    5/   10] | d_loss: 0.3110 | g_loss: 1.6645
    Epoch [    5/   10] | d_loss: 0.6087 | g_loss: 1.7842
    Epoch [    5/   10] | d_loss: 0.5965 | g_loss: 1.0506
    Epoch [    5/   10] | d_loss: 0.6210 | g_loss: 1.7927
    Epoch [    5/   10] | d_loss: 0.7243 | g_loss: 3.9066
    Epoch [    5/   10] | d_loss: 0.6715 | g_loss: 2.7507
    Epoch [    5/   10] | d_loss: 0.5632 | g_loss: 2.4817
    Epoch [    5/   10] | d_loss: 0.6050 | g_loss: 1.5578
    Epoch [    5/   10] | d_loss: 0.3460 | g_loss: 2.9607
    Epoch [    5/   10] | d_loss: 0.6075 | g_loss: 2.7551
    Epoch [    5/   10] | d_loss: 0.8077 | g_loss: 1.5106
    Epoch [    5/   10] | d_loss: 0.4638 | g_loss: 1.6624
    Epoch [    5/   10] | d_loss: 1.2398 | g_loss: 0.6678
    Epoch [    5/   10] | d_loss: 1.2159 | g_loss: 3.4430
    Epoch [    5/   10] | d_loss: 0.6071 | g_loss: 2.7914
    Epoch [    5/   10] | d_loss: 0.4359 | g_loss: 2.0077
    Epoch [    5/   10] | d_loss: 0.7977 | g_loss: 2.0437
    Epoch [    5/   10] | d_loss: 0.4534 | g_loss: 3.0467
    Epoch [    6/   10] | d_loss: 0.6131 | g_loss: 1.6626
    Epoch [    6/   10] | d_loss: 0.4299 | g_loss: 2.2227
    Epoch [    6/   10] | d_loss: 0.3006 | g_loss: 0.7026
    Epoch [    6/   10] | d_loss: 0.1854 | g_loss: 2.6362
    Epoch [    6/   10] | d_loss: 0.5243 | g_loss: 2.2669
    Epoch [    6/   10] | d_loss: 0.6989 | g_loss: 1.7068
    Epoch [    6/   10] | d_loss: 0.4597 | g_loss: 1.4009
    Epoch [    6/   10] | d_loss: 0.9632 | g_loss: 1.0400
    Epoch [    6/   10] | d_loss: 1.3017 | g_loss: 0.9841
    Epoch [    6/   10] | d_loss: 0.4176 | g_loss: 2.6286
    Epoch [    6/   10] | d_loss: 0.4484 | g_loss: 1.8289
    Epoch [    6/   10] | d_loss: 0.4174 | g_loss: 1.4095
    Epoch [    6/   10] | d_loss: 0.7201 | g_loss: 1.3313
    Epoch [    6/   10] | d_loss: 0.2836 | g_loss: 0.5963
    Epoch [    6/   10] | d_loss: 0.3262 | g_loss: 2.7093
    Epoch [    6/   10] | d_loss: 0.5079 | g_loss: 1.6562
    Epoch [    6/   10] | d_loss: 0.5641 | g_loss: 2.6865
    Epoch [    6/   10] | d_loss: 0.4018 | g_loss: 3.5414
    Epoch [    6/   10] | d_loss: 0.7538 | g_loss: 1.3195
    Epoch [    6/   10] | d_loss: 0.6896 | g_loss: 2.3884
    Epoch [    6/   10] | d_loss: 0.5189 | g_loss: 2.3032
    Epoch [    6/   10] | d_loss: 0.9189 | g_loss: 1.8686
    Epoch [    6/   10] | d_loss: 0.4986 | g_loss: 3.0515
    Epoch [    6/   10] | d_loss: 0.3765 | g_loss: 2.6408
    Epoch [    6/   10] | d_loss: 1.4173 | g_loss: 0.8775
    Epoch [    6/   10] | d_loss: 0.5607 | g_loss: 2.1340
    Epoch [    6/   10] | d_loss: 0.8014 | g_loss: 2.4330
    Epoch [    6/   10] | d_loss: 0.5984 | g_loss: 0.9653
    Epoch [    6/   10] | d_loss: 0.6888 | g_loss: 1.7870
    Epoch [    6/   10] | d_loss: 0.2652 | g_loss: 3.0309
    Epoch [    6/   10] | d_loss: 0.4509 | g_loss: 1.8343
    Epoch [    6/   10] | d_loss: 0.8102 | g_loss: 2.9208
    Epoch [    6/   10] | d_loss: 0.5327 | g_loss: 2.4754
    Epoch [    6/   10] | d_loss: 0.4832 | g_loss: 2.7773
    Epoch [    6/   10] | d_loss: 0.5500 | g_loss: 3.3042
    Epoch [    6/   10] | d_loss: 0.5702 | g_loss: 3.2377
    Epoch [    6/   10] | d_loss: 0.7310 | g_loss: 1.1722
    Epoch [    6/   10] | d_loss: 0.5023 | g_loss: 2.2215
    Epoch [    6/   10] | d_loss: 0.8471 | g_loss: 3.6402
    Epoch [    6/   10] | d_loss: 0.5849 | g_loss: 2.6509
    Epoch [    6/   10] | d_loss: 0.3653 | g_loss: 2.6616
    Epoch [    6/   10] | d_loss: 0.3248 | g_loss: 1.9398
    Epoch [    6/   10] | d_loss: 0.9223 | g_loss: 4.0499
    Epoch [    6/   10] | d_loss: 0.6721 | g_loss: 3.2913
    Epoch [    6/   10] | d_loss: 0.7161 | g_loss: 1.5024
    Epoch [    6/   10] | d_loss: 0.4479 | g_loss: 2.5791
    Epoch [    6/   10] | d_loss: 0.5212 | g_loss: 1.7683
    Epoch [    6/   10] | d_loss: 0.5045 | g_loss: 1.3497
    Epoch [    6/   10] | d_loss: 0.6152 | g_loss: 1.1267
    Epoch [    6/   10] | d_loss: 0.5551 | g_loss: 1.5158
    Epoch [    6/   10] | d_loss: 0.4587 | g_loss: 1.8742
    Epoch [    6/   10] | d_loss: 0.6807 | g_loss: 2.9760
    Epoch [    6/   10] | d_loss: 0.5110 | g_loss: 2.5312
    Epoch [    6/   10] | d_loss: 0.8837 | g_loss: 3.1058
    Epoch [    6/   10] | d_loss: 0.3380 | g_loss: 4.4900
    Epoch [    6/   10] | d_loss: 0.6072 | g_loss: 3.4840
    Epoch [    6/   10] | d_loss: 0.4818 | g_loss: 0.8018
    Epoch [    7/   10] | d_loss: 0.3787 | g_loss: 3.2713
    Epoch [    7/   10] | d_loss: 0.4111 | g_loss: 2.1725
    Epoch [    7/   10] | d_loss: 0.6174 | g_loss: 2.1986
    Epoch [    7/   10] | d_loss: 0.4450 | g_loss: 2.6382
    Epoch [    7/   10] | d_loss: 0.3990 | g_loss: 3.3629
    Epoch [    7/   10] | d_loss: 0.3088 | g_loss: 2.7460
    Epoch [    7/   10] | d_loss: 0.4958 | g_loss: 1.6353
    Epoch [    7/   10] | d_loss: 0.6304 | g_loss: 2.3534
    Epoch [    7/   10] | d_loss: 0.2532 | g_loss: 4.5817
    Epoch [    7/   10] | d_loss: 0.6479 | g_loss: 3.0984
    Epoch [    7/   10] | d_loss: 0.5513 | g_loss: 3.4089
    Epoch [    7/   10] | d_loss: 0.3134 | g_loss: 2.6342
    Epoch [    7/   10] | d_loss: 0.3460 | g_loss: 1.7633
    Epoch [    7/   10] | d_loss: 0.6270 | g_loss: 1.3131
    Epoch [    7/   10] | d_loss: 0.4562 | g_loss: 1.2855
    Epoch [    7/   10] | d_loss: 0.6967 | g_loss: 3.7443
    Epoch [    7/   10] | d_loss: 0.3883 | g_loss: 1.1247
    Epoch [    7/   10] | d_loss: 1.1162 | g_loss: 1.4496
    Epoch [    7/   10] | d_loss: 0.6862 | g_loss: 1.9642
    Epoch [    7/   10] | d_loss: 0.7530 | g_loss: 2.4123
    Epoch [    7/   10] | d_loss: 0.7394 | g_loss: 0.2484
    Epoch [    7/   10] | d_loss: 0.6291 | g_loss: 1.7968
    Epoch [    7/   10] | d_loss: 1.1447 | g_loss: 0.4640
    Epoch [    7/   10] | d_loss: 0.4903 | g_loss: 1.3460
    Epoch [    7/   10] | d_loss: 0.5742 | g_loss: 2.7956
    Epoch [    7/   10] | d_loss: 0.6709 | g_loss: 2.9093
    Epoch [    7/   10] | d_loss: 0.1588 | g_loss: 3.7518
    Epoch [    7/   10] | d_loss: 0.3172 | g_loss: 2.7774
    Epoch [    7/   10] | d_loss: 0.5264 | g_loss: 2.1130
    Epoch [    7/   10] | d_loss: 0.7936 | g_loss: 2.6904
    Epoch [    7/   10] | d_loss: 0.2563 | g_loss: 1.9693
    Epoch [    7/   10] | d_loss: 0.6558 | g_loss: 1.7162
    Epoch [    7/   10] | d_loss: 0.3564 | g_loss: 1.8198
    Epoch [    7/   10] | d_loss: 0.4630 | g_loss: 2.8980
    Epoch [    7/   10] | d_loss: 0.8806 | g_loss: 0.8306
    Epoch [    7/   10] | d_loss: 0.3239 | g_loss: 1.6085
    Epoch [    7/   10] | d_loss: 0.4486 | g_loss: 2.5150
    Epoch [    7/   10] | d_loss: 0.8813 | g_loss: 1.4289
    Epoch [    7/   10] | d_loss: 0.3901 | g_loss: 3.3365
    Epoch [    7/   10] | d_loss: 0.1809 | g_loss: 2.6817
    Epoch [    7/   10] | d_loss: 0.3283 | g_loss: 1.8781
    Epoch [    7/   10] | d_loss: 0.7115 | g_loss: 2.0561
    Epoch [    7/   10] | d_loss: 0.4457 | g_loss: 2.0092
    Epoch [    7/   10] | d_loss: 0.6913 | g_loss: 2.0837
    Epoch [    7/   10] | d_loss: 0.3181 | g_loss: 1.7014
    Epoch [    7/   10] | d_loss: 0.4084 | g_loss: 1.0058
    Epoch [    7/   10] | d_loss: 0.2859 | g_loss: 2.8504
    Epoch [    7/   10] | d_loss: 0.2003 | g_loss: 2.5487
    Epoch [    7/   10] | d_loss: 1.1387 | g_loss: 1.0666
    Epoch [    7/   10] | d_loss: 0.6104 | g_loss: 2.0029
    Epoch [    7/   10] | d_loss: 0.4825 | g_loss: 2.1227
    Epoch [    7/   10] | d_loss: 0.4514 | g_loss: 2.6752
    Epoch [    7/   10] | d_loss: 1.2751 | g_loss: 2.3475
    Epoch [    7/   10] | d_loss: 0.8994 | g_loss: 3.8799
    Epoch [    7/   10] | d_loss: 0.3868 | g_loss: 2.2264
    Epoch [    7/   10] | d_loss: 0.3284 | g_loss: 1.9207
    Epoch [    7/   10] | d_loss: 0.3513 | g_loss: 2.3701
    Epoch [    8/   10] | d_loss: 0.3545 | g_loss: 1.5507
    Epoch [    8/   10] | d_loss: 0.2104 | g_loss: 3.5212
    Epoch [    8/   10] | d_loss: 0.5248 | g_loss: 2.2367
    Epoch [    8/   10] | d_loss: 0.4900 | g_loss: 4.0762
    Epoch [    8/   10] | d_loss: 0.4916 | g_loss: 3.2304
    Epoch [    8/   10] | d_loss: 0.2471 | g_loss: 1.7792
    Epoch [    8/   10] | d_loss: 0.6362 | g_loss: 3.2936
    Epoch [    8/   10] | d_loss: 0.5648 | g_loss: 2.2958
    Epoch [    8/   10] | d_loss: 0.3065 | g_loss: 3.9343
    Epoch [    8/   10] | d_loss: 0.5667 | g_loss: 2.6154
    Epoch [    8/   10] | d_loss: 0.2314 | g_loss: 1.7458
    Epoch [    8/   10] | d_loss: 0.2667 | g_loss: 2.6632
    Epoch [    8/   10] | d_loss: 1.0024 | g_loss: 3.9373
    Epoch [    8/   10] | d_loss: 0.2404 | g_loss: 1.8878
    Epoch [    8/   10] | d_loss: 0.6905 | g_loss: 2.8458
    Epoch [    8/   10] | d_loss: 0.2784 | g_loss: 2.5856
    Epoch [    8/   10] | d_loss: 0.5212 | g_loss: 2.4392
    Epoch [    8/   10] | d_loss: 0.4670 | g_loss: 2.4646
    Epoch [    8/   10] | d_loss: 1.0045 | g_loss: 4.2842
    Epoch [    8/   10] | d_loss: 0.2298 | g_loss: 4.3155
    Epoch [    8/   10] | d_loss: 0.2212 | g_loss: 2.6108
    Epoch [    8/   10] | d_loss: 0.3668 | g_loss: 2.5877
    Epoch [    8/   10] | d_loss: 0.7712 | g_loss: 3.1950
    Epoch [    8/   10] | d_loss: 0.3119 | g_loss: 1.8850
    Epoch [    8/   10] | d_loss: 0.2629 | g_loss: 2.6502
    Epoch [    8/   10] | d_loss: 0.2770 | g_loss: 1.8937
    Epoch [    8/   10] | d_loss: 0.4351 | g_loss: 2.0899
    Epoch [    8/   10] | d_loss: 0.2559 | g_loss: 2.8211
    Epoch [    8/   10] | d_loss: 0.9524 | g_loss: 1.7890
    Epoch [    8/   10] | d_loss: 0.3591 | g_loss: 3.1456
    Epoch [    8/   10] | d_loss: 0.5351 | g_loss: 1.7695
    Epoch [    8/   10] | d_loss: 0.9661 | g_loss: 4.8195
    Epoch [    8/   10] | d_loss: 0.4399 | g_loss: 3.0966
    Epoch [    8/   10] | d_loss: 0.7260 | g_loss: 2.5558
    Epoch [    8/   10] | d_loss: 0.3313 | g_loss: 2.9619
    Epoch [    8/   10] | d_loss: 0.4049 | g_loss: 2.5061
    Epoch [    8/   10] | d_loss: 0.7257 | g_loss: 3.5069
    Epoch [    8/   10] | d_loss: 0.5488 | g_loss: 2.5658
    Epoch [    8/   10] | d_loss: 0.5171 | g_loss: 2.5113
    Epoch [    8/   10] | d_loss: 0.2932 | g_loss: 2.5040
    Epoch [    8/   10] | d_loss: 0.4522 | g_loss: 2.6739
    Epoch [    8/   10] | d_loss: 0.5376 | g_loss: 2.9858
    Epoch [    8/   10] | d_loss: 0.5467 | g_loss: 1.6005
    Epoch [    8/   10] | d_loss: 0.3488 | g_loss: 2.6331
    Epoch [    8/   10] | d_loss: 0.5938 | g_loss: 0.8402
    Epoch [    8/   10] | d_loss: 0.1440 | g_loss: 1.7671
    Epoch [    8/   10] | d_loss: 0.4936 | g_loss: 2.6844
    Epoch [    8/   10] | d_loss: 0.7796 | g_loss: 1.9666
    Epoch [    8/   10] | d_loss: 0.3242 | g_loss: 3.2707
    Epoch [    8/   10] | d_loss: 0.3156 | g_loss: 2.2630
    Epoch [    8/   10] | d_loss: 1.4755 | g_loss: 1.0196
    Epoch [    8/   10] | d_loss: 0.3777 | g_loss: 2.7318
    Epoch [    8/   10] | d_loss: 0.3318 | g_loss: 2.6214
    Epoch [    8/   10] | d_loss: 0.6964 | g_loss: 1.4799
    Epoch [    8/   10] | d_loss: 0.6144 | g_loss: 3.8209
    Epoch [    8/   10] | d_loss: 0.5881 | g_loss: 2.3210
    Epoch [    8/   10] | d_loss: 0.3095 | g_loss: 3.3764
    Epoch [    9/   10] | d_loss: 0.7958 | g_loss: 2.0137
    Epoch [    9/   10] | d_loss: 1.1180 | g_loss: 4.9843
    Epoch [    9/   10] | d_loss: 0.3233 | g_loss: 2.2818
    Epoch [    9/   10] | d_loss: 0.4081 | g_loss: 1.6872
    Epoch [    9/   10] | d_loss: 0.1328 | g_loss: 1.7398
    Epoch [    9/   10] | d_loss: 0.3839 | g_loss: 3.1949
    Epoch [    9/   10] | d_loss: 0.3956 | g_loss: 3.4710
    Epoch [    9/   10] | d_loss: 0.4662 | g_loss: 2.0692
    Epoch [    9/   10] | d_loss: 0.1396 | g_loss: 2.1416
    Epoch [    9/   10] | d_loss: 0.7033 | g_loss: 4.5267
    Epoch [    9/   10] | d_loss: 0.3810 | g_loss: 2.4842
    Epoch [    9/   10] | d_loss: 0.8668 | g_loss: 3.4257
    Epoch [    9/   10] | d_loss: 0.8092 | g_loss: 2.0416
    Epoch [    9/   10] | d_loss: 0.1970 | g_loss: 2.4944
    Epoch [    9/   10] | d_loss: 0.3658 | g_loss: 1.3878
    Epoch [    9/   10] | d_loss: 0.5938 | g_loss: 4.3212
    Epoch [    9/   10] | d_loss: 0.6824 | g_loss: 3.0047
    Epoch [    9/   10] | d_loss: 0.4710 | g_loss: 2.3355
    Epoch [    9/   10] | d_loss: 0.4070 | g_loss: 2.6194
    Epoch [    9/   10] | d_loss: 0.6451 | g_loss: 3.7926
    Epoch [    9/   10] | d_loss: 0.4403 | g_loss: 2.0283
    Epoch [    9/   10] | d_loss: 0.3301 | g_loss: 3.2711
    Epoch [    9/   10] | d_loss: 0.5004 | g_loss: 3.1164
    Epoch [    9/   10] | d_loss: 0.2670 | g_loss: 4.5862
    Epoch [    9/   10] | d_loss: 0.3445 | g_loss: 2.5394
    Epoch [    9/   10] | d_loss: 0.4404 | g_loss: 2.4993
    Epoch [    9/   10] | d_loss: 0.2074 | g_loss: 3.8929
    Epoch [    9/   10] | d_loss: 0.3747 | g_loss: 2.8142
    Epoch [    9/   10] | d_loss: 0.3066 | g_loss: 2.4967
    Epoch [    9/   10] | d_loss: 0.3258 | g_loss: 2.8470
    Epoch [    9/   10] | d_loss: 0.2147 | g_loss: 3.2469
    Epoch [    9/   10] | d_loss: 0.1356 | g_loss: 4.4160
    Epoch [    9/   10] | d_loss: 0.1353 | g_loss: 3.1965
    Epoch [    9/   10] | d_loss: 0.6778 | g_loss: 4.4086
    Epoch [    9/   10] | d_loss: 0.3097 | g_loss: 1.4800
    Epoch [    9/   10] | d_loss: 0.3600 | g_loss: 1.2397
    Epoch [    9/   10] | d_loss: 0.5439 | g_loss: 4.4019
    Epoch [    9/   10] | d_loss: 0.2376 | g_loss: 2.5645
    Epoch [    9/   10] | d_loss: 1.0602 | g_loss: 4.8398
    Epoch [    9/   10] | d_loss: 0.1502 | g_loss: 3.6295
    Epoch [    9/   10] | d_loss: 0.3918 | g_loss: 4.1729
    Epoch [    9/   10] | d_loss: 0.3563 | g_loss: 2.6311
    Epoch [    9/   10] | d_loss: 0.3923 | g_loss: 2.1126
    Epoch [    9/   10] | d_loss: 0.2049 | g_loss: 3.7298
    Epoch [    9/   10] | d_loss: 1.0741 | g_loss: 2.4022
    Epoch [    9/   10] | d_loss: 0.1391 | g_loss: 2.6011
    Epoch [    9/   10] | d_loss: 1.1179 | g_loss: 1.7936
    Epoch [    9/   10] | d_loss: 0.3378 | g_loss: 2.3964
    Epoch [    9/   10] | d_loss: 0.2668 | g_loss: 3.0656
    Epoch [    9/   10] | d_loss: 0.0952 | g_loss: 3.2094
    Epoch [    9/   10] | d_loss: 0.2538 | g_loss: 3.7583
    Epoch [    9/   10] | d_loss: 0.3544 | g_loss: 1.5199
    Epoch [    9/   10] | d_loss: 0.5133 | g_loss: 3.7610
    Epoch [    9/   10] | d_loss: 0.5193 | g_loss: 4.1392
    Epoch [    9/   10] | d_loss: 0.5395 | g_loss: 5.6215
    Epoch [    9/   10] | d_loss: 0.3722 | g_loss: 2.2540
    Epoch [    9/   10] | d_loss: 0.3496 | g_loss: 3.0168
    Epoch [   10/   10] | d_loss: 0.1506 | g_loss: 3.1284
    Epoch [   10/   10] | d_loss: 0.5005 | g_loss: 2.7923
    Epoch [   10/   10] | d_loss: 0.2329 | g_loss: 4.3172
    Epoch [   10/   10] | d_loss: 0.2920 | g_loss: 2.8481
    Epoch [   10/   10] | d_loss: 0.4869 | g_loss: 2.7364
    Epoch [   10/   10] | d_loss: 0.3645 | g_loss: 1.3832
    Epoch [   10/   10] | d_loss: 0.2958 | g_loss: 4.2701
    Epoch [   10/   10] | d_loss: 0.2752 | g_loss: 4.2040
    Epoch [   10/   10] | d_loss: 0.3000 | g_loss: 2.0245
    Epoch [   10/   10] | d_loss: 0.1832 | g_loss: 3.6009
    Epoch [   10/   10] | d_loss: 0.6361 | g_loss: 3.0735
    Epoch [   10/   10] | d_loss: 0.4378 | g_loss: 3.5753
    Epoch [   10/   10] | d_loss: 0.7674 | g_loss: 1.6762
    Epoch [   10/   10] | d_loss: 0.0906 | g_loss: 2.8359
    Epoch [   10/   10] | d_loss: 0.2643 | g_loss: 3.0888
    Epoch [   10/   10] | d_loss: 0.2865 | g_loss: 5.3736
    Epoch [   10/   10] | d_loss: 0.1423 | g_loss: 2.6674
    Epoch [   10/   10] | d_loss: 0.3027 | g_loss: 1.9178
    Epoch [   10/   10] | d_loss: 0.0606 | g_loss: 2.7456
    Epoch [   10/   10] | d_loss: 0.2992 | g_loss: 2.0044
    Epoch [   10/   10] | d_loss: 0.0929 | g_loss: 2.7819
    Epoch [   10/   10] | d_loss: 0.5342 | g_loss: 3.5143
    Epoch [   10/   10] | d_loss: 0.5569 | g_loss: 2.7988
    Epoch [   10/   10] | d_loss: 0.5476 | g_loss: 4.1120
    Epoch [   10/   10] | d_loss: 0.4387 | g_loss: 3.9358
    Epoch [   10/   10] | d_loss: 0.2141 | g_loss: 3.2421
    Epoch [   10/   10] | d_loss: 0.3969 | g_loss: 2.3368
    Epoch [   10/   10] | d_loss: 0.6126 | g_loss: 2.3273
    Epoch [   10/   10] | d_loss: 0.3728 | g_loss: 4.0061
    Epoch [   10/   10] | d_loss: 0.1637 | g_loss: 2.8562
    Epoch [   10/   10] | d_loss: 0.3026 | g_loss: 4.2510
    Epoch [   10/   10] | d_loss: 0.9926 | g_loss: 1.6901
    Epoch [   10/   10] | d_loss: 0.4880 | g_loss: 3.0351
    Epoch [   10/   10] | d_loss: 0.5228 | g_loss: 1.2235
    Epoch [   10/   10] | d_loss: 0.8631 | g_loss: 5.2191
    Epoch [   10/   10] | d_loss: 0.2152 | g_loss: 5.0519
    Epoch [   10/   10] | d_loss: 0.3977 | g_loss: 3.8114
    Epoch [   10/   10] | d_loss: 2.2134 | g_loss: 4.7074
    Epoch [   10/   10] | d_loss: 0.3417 | g_loss: 2.9938
    Epoch [   10/   10] | d_loss: 0.9200 | g_loss: 2.3087
    Epoch [   10/   10] | d_loss: 0.4359 | g_loss: 1.9508
    Epoch [   10/   10] | d_loss: 0.2884 | g_loss: 3.0795
    Epoch [   10/   10] | d_loss: 0.1838 | g_loss: 3.2346
    Epoch [   10/   10] | d_loss: 0.1815 | g_loss: 2.5961
    Epoch [   10/   10] | d_loss: 0.3024 | g_loss: 3.3063
    Epoch [   10/   10] | d_loss: 0.6810 | g_loss: 2.7535
    Epoch [   10/   10] | d_loss: 0.6004 | g_loss: 3.7914
    Epoch [   10/   10] | d_loss: 0.3055 | g_loss: 4.7553
    Epoch [   10/   10] | d_loss: 0.3839 | g_loss: 2.9566
    Epoch [   10/   10] | d_loss: 0.5230 | g_loss: 2.2097
    Epoch [   10/   10] | d_loss: 0.1586 | g_loss: 1.2258
    Epoch [   10/   10] | d_loss: 0.5196 | g_loss: 3.2109
    Epoch [   10/   10] | d_loss: 0.1643 | g_loss: 3.9196
    Epoch [   10/   10] | d_loss: 0.1626 | g_loss: 2.8338
    Epoch [   10/   10] | d_loss: 0.1378 | g_loss: 2.6072
    Epoch [   10/   10] | d_loss: 0.3793 | g_loss: 4.5483
    Epoch [   10/   10] | d_loss: 0.2544 | g_loss: 3.4693
    

## Training loss

Plot the training losses for the generator and discriminator, recorded after each epoch.


```python
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f13f004a128>




    

![png](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/output_38_1.png)
    


## Generator samples from training

View samples of images from the generator, and answer a question about the strengths and weaknesses of your trained models.


```python
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
```


```python
# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
```


```python
_ = view_samples(-1, samples)
```


    
![png](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/output_42_0.png)
    


### Question: What do you notice about your generated samples and how might you improve this model?
When you answer this question, consider the following factors:
* The dataset is biased; it is made of "celebrity" faces that are mostly white
* Model size; larger models have the opportunity to learn more features in a data feature space
* Optimization strategy; optimizers and number of epochs affect your final result


**Answer:** 
1. At the end of 10 iteration,it's clear that the discriminator is performing better than the generator.The facial features are a bit complex even for a model with 4 convolutions.
2. Increasing the convolution layers and the no. of epochs for the current setting can give better results.
3. I'm a bit hesistant to tweak the hyperparameters since the gans are sensitive to hyperparameters so I went with the default parameters mentioned in the paper.
4. Using a different optimizer could also produce better results.
