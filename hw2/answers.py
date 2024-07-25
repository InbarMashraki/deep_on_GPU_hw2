r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 1
    lr = 0.05
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 3
    lr_vanilla = 0.07
    lr_momentum = 0.001
    lr_rmsprop = 0.0001
    reg = 0.0005
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.0025
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr=0.001
    weight_decay=0.001
    momentum= 0.999
    loss_fn= torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1. Number of Parameters
- Regular Block: The regular block has two $3 \times 3$ convolutional layers. Input size: 64d, Output size: 64d for both layers.

the number of parameters for each of the $3 \times 3$ Conv Layers: (3 * 3 * 64 + 1) * 64 = 576 * 64 = 36864. 

there for the number of parameters is 36864 * 2 = 73728

the shortcut is the identity function and has no parameters.


- Bottleneck Block: The bottleneck block has three convolutional layers: $1 \times 1$, $3 \times 3$, and $1 \times 1$. size: 256, Output size: 256, with intermediate size: 64.

First $1 \times 1$ Conv Layer : (1 * 1 * 256 + 1) * 64 = 256 * 64 + 64 = 16448

Second $3 \times 3$ Conv Layer : (3* 3 * 64 + 1) * 64 = 576 * 64 = 36864

Third $1 \times 1$ Conv Layer : (1 * 1 * 64 + 1) * 256 = 64 * 256 + 256 = 16640

there for the number of parameters is 16448 + 36864 + 16640 = 69952

comparison- The bottleneck block has significantly more parameters (35072) compared to the regular block (4608).

2. Number of Floating Point Operations

since we are saving the spacial dimantions of the input by strides and padding, for all the convolutions layers, both in the regular block and the bottleneck one, 
the input size is H * W * (in_channels)- where W,H are the input width and height.
For each output pixel, the convolution operation is applied, involving multiplications and additions over all input channels and the entire kernel size.
So, for a given convolutional layer, the FLOPs can be calculated as: H_{out}* W_{out}* C_{out} \times (k * k * C_{in})

- Regular Block: for both convolutional layers in the main path, the input size and the output size is (H,W,64). 
Also for both of the convolutional layers, the kernel size is 3 \times 3.

So, the number of Floating Point Operations is 2 * (H * W * 64) *(3 * 3 * 64) = H * W * 73728

- Bottleneck Block:

First $1 \times 1$ Conv Layer : (H * W * 64) * (1 * 1 * 256)= H * W * 16384

Second $3 \times 3$ Conv Layer : (H * W * 64) * (3 * 3 * 64)= H * W * 36864

Third $1 \times 1$ Conv Layer : (H * W * 256) * (1 * 1 * 64)= H * W * 16384

So, the number of Floating Point Operations is 2 * H * W * 16384 + H * W * 36864 = H * W * 69632 

Comparison- The bottleneck block requires fewer FLOPs (69632 per spatial location) compared to the regular block (73728 per spatial location),
making it more efficient in terms of computation.

3. Ability to combine the input: 

(1) spatially (within feature maps)
- Regular Block:
Each $3 \times 3$ convolution has a receptive field of $3 \times 3$ pixels. When stacked, the two $3 \times 3$ convolutions effectively have a receptive field of $5 \times 5$ pixels,because the second layer can see a bit further into the input through the output of the first layer.
This means that each output pixel has access to information from a $5 \times 5$ region of the input, allowing for effective spatial feature extraction and local pattern recognition. The shortcut identity path bypasses the convolutions, allowing the network to combine learned spatial features with the original input spatial features directly, preserving important spatial information.

- Bottleneck Block:
The $3 \times 3$ convolution in the middle has a receptive field of $3 \times 3$ pixels. The overall receptive field of the bottleneck block is effectively still $3 \times 3$, because the $1 \times 1$ convolutions do not increase the spatial receptive field. However, the $1 \times 1$ convolutions before and after the $3 \times 3$ layer allow for more flexible spatial feature extraction by projecting the high-dimensional input (256 channels) down to a lower-dimensional space (64 channels) and then back up.
The $1 \times 1$ convolution in the shortcut path matches the dimensions of the original and learned features, allowing them to be combined effectively and preserving spatial information.

(2) across feature maps
- Regular Block:

First Convolutional Layer: Each $3 \times 3$ filter processes all 64 input channels to produce one output channel. This means each output channel combines information from all 64 input channels through local $3 \times 3$ regions.
With 64 filters, this layer produces 64 output channels.
The ReLU activation applied after this convolution introduces non-linearity, allowing the network to model more complex relationships. 

Second Convolutional Layer:
This layer behave in the exact same way as the first layer.

So, the regular block allows for combination of channel information at each layer, with each convolution combining information from all input channels. 
However, since the number of input and output channels is the same, the ability to mix and combine information across channels is somewhat limited.

The shortcut identity path provides a direct channel for the original feature maps to be added to the output, which helps in preserving and integrating feature information across channels.

- Bottleneck Block:

First $1 \times 1$ Convolutional Layer:
Reduces the dimensionality from 256 input channels to 64 channels. Each output channel is a combination of all 256 input channels.
The ReLU activation applied here introduces non-linearity, allowing for complex mappings from the high-dimensional input space to the lower-dimensional space.

Second $3 \times 3$ Convolutional Layer:
Operates on the 64 intermediate channels, combining information across these channels with a local receptive field of $3 \times 3$ pixels.

Third $1 \times 1$ Convolutional Layer:
Expands the dimensionality back to 256 channels. Each output channel is a combination of the 64 intermediate channels.

So, The bottleneck block enhances the ability to combine information across channels by using $1 \times 1$ convolutions, which provide a more flexible and efficient channel-wise mixing. 

The shortcut identity path matches the dimensionality of the output from the main path, allowing the combination of original high-dimensional features with the refined features, improving feature integration across channels.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
1. Model Performance and Confidence:
IMPORTANT NOTE: In the following images, we chose images containing objects that can be detect by the model, meaning the model has a label class for this object.
By doing so, we make sure that the model fails due to the conditions of the image and not because lack of knowlage. 
image1:

The YOLOv5 model detected 3 bounding boxes in the image:

Bounding Box 1: Detected as a person with a confidence of 0.47 (actually a dolphin).
Bounding Box 2: Detected as a person with a confidence of 0.90 (actually 2 dolphins).
Bounding Box 3: Detected as a surfboard with a confidence of 0.67 (actually a tail of one of the dolphins).

image2:
The YOLOv5 model detected 3 bounding boxes in the image:

Bounding Box 1: Detected as a cat with a confidence of 0.65 (actually a dog and part of a cat).
Bounding Box 2: Detected as a cat with a confidence of 0.39 (actually a dog).
Bounding Box 3: Detected as a dog with a confidence of 0.50 (overlapping another dog).

2.Possible Reasons for Model Failures: 
    *Overlapping Objects and Similar Fur Colors:*
    When objects overlap significantly and have similar appearances or textures, the model might have difficulty distinguishing between them.
    *Silhouettes and Low Contrast:*
    Shadows and silhouettes lack distinct features, and low contrast between objects and background can make detection difficult.

Methods to Resolve These Issues:
    *Data Augmentation:* Use techniques such as contrast adjustment to expose the model to a wide range of lighting scenarios, including shadows and silhouettes. This helps the model learn to recognize objects based on shape and context rather than just texture and color.
    *Dataset Enhancement:* Include more images with overlapping objects and similar textures in the training dataset to better represent real-world scenarios. This improves the model's ability to distinguish between objects with similar appearances.
   
3.To carry out an adversarial attack on a YOLO object detection model, we would start by creating a clone of the input image and setting it to require gradients.
We would then iteratively update the adversarial image copy to maximize the loss (rather than minimize it) by taking gradient steps. The objective is to maximize the loss of the object detection model, causing it to misclassify objects, fail to detect objects, or create false positives.
 After each step, we would project the perturbations back to ensure they remain within a specified epsilon norm limit. The goal is to find a small perturbation on a certain input, in a way that is almost imperceptible to humans but causes the model to make incorrect predictions.
                    
                """


part6_q2 = r"""
LALA

"""


part6_q3 = r"""
Model Performance and Confidence:

image3:

The YOLOv5 model detected 3 bounding boxes in the image:

Bounding Box 1: Detected as a microwave correctly but with a confidence of 0.66 .
Bounding Box 2: Detected a bowl as a cup with a confidence of 0.73. 
Bounding Box 3: Detected several plates as a cup with a confidence of 0.43.
The oven in the image was not detected!

important: the model does not have a 'bowl' or a 'plate' class. 

The model performed poorly in detecting the oven in the image due to lighting conditions and blur.

image4:
The YOLOv5 model did not detect the car in the image because it was partially occluded, thus missing important features.

image5:
The YOLOv5 model detected 3 bounding boxes in the image:

Bounding Box 1: Detected a carrot as a spoon but with a confidence of 0.25.
Bounding Box 2: Detected as a cup correctly with a confidence of 0.55.

The YOLOv5 model exhibited model bias by mistaking the carrot for a spoon due to the context. 
This might be because the model has learned to associate objects in specific setups, resulting in incorrect identification in this unusual arrangement.

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""