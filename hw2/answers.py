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
    momentum= 0.99
    loss_fn= torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
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