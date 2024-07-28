r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
1. 
   A. To understand the shape of the jacobian of the fully connected layer with respect to x:
  $dy/dx$
  let us first recall the shape of each component of the fully connected layer $Y=WX$:
- X: `N x in_features`
- W: `in_features x out_features`
- Y: `N x out_features` \
  Now recall that the jacobian holds the derivatives of each output cell of $Y$ with respect to each of the input cells of $X$, therefor the shape of the jacobian will be: \
  `N x out_features x N x in_features = 64 x 512 x 64 x 1024`.

  B. The Jacobian is indeed sparse.
  Recall the Jacobian[`i,j,u,v`] is calculated by Deriving $Y_{i,j}$ with respect to $X_{u,v}$, while $Y_{i,j}$ is affected only by the $i$th sample in batch, therefor the value of Jacobian[`i,j,u,v`] will be $0$ for each $i \ne u$, which is very common case in the Jacobian, therefor it is indeed sparse.
  More precisely, $Y_{i,j}$ is calculated as:\
  $Y_{i,k}=\sum_{j=1}^{in-features}x_{i,j}\cdot w_{j,k}$, we can clearly see that $Y_{i,k}$ is is built from $x_i$, therefor any derivates of  $Y_{i,k}$ with respect to $X_{u \ne i}$ will be $0$.
  
  As we shown, the shape of $Y$ is
  `N x out_features`, the `N` is for each sample in the batch

  C. There is no need to materialize the entire Jacobian tensor for the gradient calculation. Generally, from the chain rule we know that the gradient is $\frac{dL}{dX}=\frac{dL}{dY} \cdot \frac{dY}{dX}$, since $Y=WX$ we can say that
  $\frac{dL}{dX}=\frac{dL}{dY} \cdot \frac{d(WX)}{dX}=
  \frac{dL}{dY} \cdot W^T$ and since $\frac{dL}{dY}$ is given we can now completely compute $\frac{dL}{dX}$ without materializing the entire Jacobian.

2.
    A. With the exact same explanation part A and given the shapes of all relevant tensors that we written in part A, the shape of the Jacobian tensor of $Y$ with respect to $W$ will be: \
    `N x out_features x in_features x out_features = 64 x 512 x 1024 x 512`.

    B. As we have shown in part 1.B, $Y_{i,k}=\sum_{j=1}^{in-features}x_{i,j}\cdot w_{j,k}$, we can clearly see that $Y_{i,k}$ is is built from $w_k$, therefor any derivatives of $Y_{i,k}$ with respect to $W_{j, u \ne k}$ will be $0$. Therefor the Jacobian of $Y$ w.r.t $W$ is sparse.

    C. Just as in 1.C, There is no need to materialize the entire Jacobian tensor for the gradient calculation. Generally, from the chain rule we know that the gradient is $\frac{dL}{dX}=\frac{dL}{dY} \cdot \frac{dY}{dW}$, since $Y=WX$ we can say that
  $\frac{dL}{dX}=\frac{dL}{dY} \cdot \frac{d(WX)}{dW}=
  \frac{dL}{dY} \cdot X^T$ and since $\frac{dL}{dY}$ is given we can now completely compute $\frac{dL}{dW}$ without materializing the entire Jacobian.

"""

part1_q2 = r"""
**Your answer:**
Let us recall that back-propagation is an algorithm to calculate function gradients using the chain rule and starting from the top of the computational graph and going backwards to it's beginning. \
Let us recall that training neural network with decent-based optimization method means we calculate gradients of the classification function with respect to model parameters and use it to try to reach a local minima of the function. Back-propagation is generally used to calculate the gradients while training NN but it is not entirely required, as we have seen in tutorial 5, we can also calculate gradients using forward automatic differentiation instead of reverse automatic differentiation (back-propagation), this is called forward mode AD. \
In forward mode AD we start the gradient calculation in the begging of the computational graph and work out way up (still by using the chain rule), as we have seen in tutorial 5 - using forward mode AD will be less efficient in most cases for training a NN and that's the reason we will use back-propagation, but it is still possible to use it for training.

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
1. The dropout vs. no-dropout graphs definitely matches our expectations. Generally we know that dropout is technique used to avoid overfitting and create a more robust model that will handle new samples better, more generalized. This is exactly what we see in the graphs, the no-dropout model reaches almost 80% accuracy on train set while only 23% on the test - this is clearly an overfitted model, on the other hand we can see both dropout models reached worse training accuracy but much much  better test accuracy - meaning those model were able to generalize the problem better, as we expected from what we know about dropout.

2. As we explained before, dropout is used to prevent overfitting, we can see that between the 2 dropout models, the one with the low-dropout was a bit overfitted (54 on train, 27.5 on test) while the high-dropout was not overfitted at all (35 on train, 30 on test). This matches the assumption that dropout prevents overfitting.
"""

part2_q2 = r"""
It is possible for the test loss to increase while the test accuracy also increase when using cross entrophy loss, this is due to the fact that cross entropy loss is sensitive to the confidence of the model's predictions - it penalizes incorrect predictions more heavily if the model is very confident in its wrong prediction. During training the model might be very insecure on predictions and also make a lot of wrong predictions, this will cause bad accuracy and medium (not so large) loss, then, after some learning the model might get more confident on predictions (both right and wrong ones) and also have some more correct predictions, this will cause better accuracy commpered to the beginning but may also cause worse (larger) accuracy. In that case both loss and accuracy will increase.
"""

part2_q3 = r"""
1. Both gradient descent and back-propagation are essential part of training deep neural networks but work on a different level of the training process.\
Gradient descent is an iterative optimization algorithm that iteratively update function parameters on the opposite direction of the gradient in order to converge to some local minima of the function, generally it doesn't specify how the gradient is calculated but just mention its usage.\
On the other hand, back-propagation is an algorithm that computes the gradient of a function (in the case of neural network - the cost function). Its implementation is based on the chain rule and the computational graph, generally in back-propagation we move backwards in the computational graph and in each step calculating relevant gradient and multiplying (according to chain rule). Back-propagation allows efficient gradient calculation and is used in most on the neural networks implementations.

2. Both gradient descent and stochastic gradient descent are algorithms that use the cost function gradients to change the parameters in order to reach the minima.
In deep learning the cost function takes points from the dataset as arguments for calculation, in GD, each iteration we take the entire dataset and calculate the average of the cost function and gradient over it. On the other hand in SGD we take a random sample out of the dataset and do the gradient calculation over it.
After initializing some values to the functions' parameters, GD is deterministic since it alway takes the entire dataset, while SGD isn't since it takes a different sample every iteration. This causes GD to be more predictable and converge more smoothly while SGD may lack those. This may also cause GD to be a bit more accurate since it is more likely to reach the actual local minima than SGD.
On the other hand, GD is obviously much slower to calculate, and when working on large dataset may not be realistic for usage while each iteration of SGD is very fast to compute.
To sum up we can say SGD is better for large datasets due to its speed and memory efficiency, while GD is better for small datasets that prioritize accuracy and predictable behavior.

3. SGD is more used in deep learning because it fits better large scale problems with large datasets. \
In modern deep learning most problems use very large dataset, ones that make gradient computation for the whole dataset very expensive in time and memory, also the cost function in deep learning can be complex and to derive over entire dataset each iteration, therefor using the entire dataset (such as they do in GD) is not practical id deep learning.

4. A. Yes, splitting the data into disjoint batches, doing multiple forward passes until all data is exhausted, and then do one backward pass on the sum of the losses should theoretically be equivalent to GD (with a small difference), generally this is true because of the fact that the sum of the gradients equal to the gradient of the sum. Let us explain further. \
Given our model use some classification function $f(x)$ and some loss function $L$ we can say that in GD our actual cost function that will be derived to get the gradient in a training iteration will be $J(\theta) = \frac{1}{N} \sum_{i=1}^{N}L(f(x_i, \theta), y_i)$. \
On the other hand if we separate the data into $B$ batches our cost function will be the sum of the cost over all batches, so we get
$J(\theta) = \sum_{b=1}^{B} \frac{1}{N/B} \sum_{i=1}^{N/B}L(f(x_{bi}, \theta), y_{bi})=\frac{1}{N/B} \sum_{i=1}^{N}L(f(x_i, \theta), y_i)$ \
We got almost the same final cost function with those 2 methods, only in the second method we multiply by different constant value, this should not be a problem because one can tune the learning rate accordingly. With almost the exact same cost function it is easy to understand that the gradients will also be equivalent, again, since the gradient of sum is sum of gradients.

B. When doing multiple forward passes the way that was introduced above, the results are summed and all the intermediate activations and computational graphs for each batch are retained in memory, which can be a lot in a deep network with large training dataset. A different approach to fix this issue can be accumulating the gradients instead - meaning after each batch forward pass we do backwards pass, save the gradient and sum it we the gradient of the next batch backwards pass gradient result.

"""

part2_q4 = r"""
1. Given function $f=f_n \circ f_{n-1} \circ ... \circ f_1$ s.t $f_i:R \rightarrow R$ is a differentiable function which is easy to evaluate and differentiate (each query costs $O(1)$ at a given point). We want to calculate $\nabla f(x_0)$ with lowest memory consumption Assuming that we are given $f$ already expressed as a computational graph.
A. Let us use forward mode AD to calculate $\nabla f(x_0)$:
As defined by forward mode AD, we will use the chain rule, starting from the beginning of the computational graph. \
As we have seen in class, let $v_i$ be the $i$th node in the computational graph, we first initialize $v_0.grad=1$ then the step is: $v_{j+1}.grad ← f'_{j+1}(v_j.val)\cdot v_j.grad$,
since we can get all derivatives of every $f_i$ in $O(1)$ each steps is done by $O(1)$, also all we need to save in memory is $v_{j+1}.grad$, therefor the memory complexity of this calculation is $O(1)$ for each node.

B. Let us use reverse mode AD (a.k.a back-propagation) to calculate $\nabla f(x_0)$:
As defined by reverse mode AD, we will use the chain rule, starting from the end of the computational graph. \
As we have seen in class, let $v_i$ be the $i$th node in the computational graph, we first initialize $v_n.grad=1$ then the step is: $v_{j-1}.grad ← f'_{j}(v_{j-1}.val)\cdot v_j.grad$,
since we can get all derivatives of every $f_i$ in $O(1)$ each steps is done by $O(1)$, also all we need to save in memory is $v_{j+1}.grad$, therefor the memory complexity of this calculation is $O(1)$.

2. To generalize these techniques to an arbitrary computational graph we have to remember that in our case all functions were 1-dimensional, when dealing with functions that get more than one scalar argument or produce more than one scalar output each node has to store the gradient w.r.t all the input nodes in forward mode or all the output nodes in reverse mode.

3. Computing gradients in big deep architectures can be benefit from back-propagation with the techniques above because the deeper the network - the more gradients are needed to be calculated (since back-prop uses the chain rule) therefor cutting memory usage can be very beneficial for networks of high depth.
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

High optimization error refers to the model's poor performance on training data, indicating that it hasn't fit the data well.
Optimization error arises often due to factors like inadequate training duration, an inappropriate learning rate, or poor optimization algorithms. 
This error is about how well we can minimize the loss function during training.
To reduce this error, we can adjust the learning rate maualy or use techniques like learning rate decay. 
We can also ensure the model is trained for a sufficient number of epochs. 
Still we need to be carefull and apply regularization techniques like dropout, weight decay, or batch normalization to prevent overfitting to the training data.

High Generalization Error indicates that the model performs well on training data but poorly on unseen test data.  
This is often due to overfitting but can also be caused by insufficient training data diversity or a complex model relative to the amount of data.
To address this, we can increase the diversity of the training data through augmentation techniques, which can help the model learn more general features.
We can also apply regularization methods such as dropout or early stopping to prevent overfitting. 
Additionally, simplifying the model or using techniques like cross-validation to find better hyper-parameters can also ensure better generalization to new data.

High approximation error occurs when the model is too simple to capture the underlying data patterns, often due to inadequate network architecture or insufficient feature representation,
leading to underfitting. This error reflects how well the model approximates the true data distribution.
To address this, we can increase the model's complexity by adding more layers or neurons, which allows it to capture more complex patterns. 
Enhancing feature representation or using advanced models, such as adjusting the receptive field in convolutional layers by increasing the kernel size, can also help. 
Experimenting with different architectures, including deeper networks or more neurons per layer, can improve the model's ability to capture more features and perform better on the training data.
"""

part3_q2 = r"""

Higher False Positive Rate (FPR):
This occurs when the classifier is set to be very sensitive to identify positive cases, like in a rare disease screening where we want to ensure no cases are missed, even if it means incorrectly labeling some healthy individuals as positive.

Higher False Negative Rate (FNR): 
This happens when the classifier is more cautious to avoid false positives, such as in spam filtering where the priority is to avoid misclassifying important emails as spam, even if it means some spam emails are missed.
"""

part3_q3 = r"""
The ROC (Receiver Operating Characteristic) curve is a graphical representation used to evaluate the performance of a binary classification model. 
It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

Scenario 1: Non-Lethal Symptoms
Sinece in this case, the person with the disease will develop non-lethal symptoms that confirm the diagnosis before treatment is needed, the primary focus is to minimize
 the cost and risks associated with further testing. 

In this scenario, it would be wise to Handle positive cases as follows:
For those flagged as positive by the low-cost screening, consider waiting for non-lethal symptoms to appear before proceeding with the high-risk, 
high-cost confirmatory tests. This approach helps reduce unnecessary high-cost tests while still ensuring that the disease will be confirmed when symptoms develop.

If it is important to detect the sick individuals early anyways, our focus shifts towards minimizing the number of unnecessary high-risk, high-cost follow-up tests, which relates to 
managing the False Positive Rate. So we would choose a point in the ROC curve with a low FPR. 
This ensures that fewer individuals are incorrectly classified as sick, thus reducing the number of people sent for unnecessary expensive and risky tests.


Scenario 2: High Risk of Death
Here, the disease is life-threatening if not detected early, and the expensive tests are the only method to confirm the diagnosis. Given this situation:

It is crucial to minimize FNR because missing a true case could result in a high risk of death. The cost and risk of the follow-up tests are secondary to ensuring that no critical cases
are missed. We would choose a point on the ROC curve that emphasizes a low FNR, even if it means a higher FPR. 
The priority is to catch all possible cases early, minimizing the risk of death, even though it might lead to more expensive follow-up tests 
but to a point that balances the high-risk to healthy patient that was tested positive.

"""


part3_q4 = r"""
Using a Multi-Layer Perceptron (MLP) for classifying the sentiment of a sentence is not ideal because MLPs treat each word independently and do not capture the sequential dependencies and
context crucial for understanding sentiment. In addition, MLPs require a fixed input size, leading to inefficiencies and potential loss of information for variable-length sentences.
More over, sentiment classification relies on the order and relationship of words, which MLPs fail to handle effectively. 
Models like RNNs, LSTMs, and Transformers are better suited as they are designed to process sequential data, capturing the necessary temporal and contextual information for accurat sentiment analysis.
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
In this experiment  we train diffrent models with different depth and kernel size. 
\\
Generally the results of experiment 1.1 were successful for L=2,4, in both cases we managed to get test accuracy of 65-70% for both k=32 and K=64. \\
1. We can see in our experiments that L=4 bring the best empirical results - both with K=32 and K=64. 
We can see in the graphs that both L=4 and L=2 learn the data and increase accuracy but in L=2 the test accuracy converges before the L=4 models, that may be due to its limitations, with small amount of layers the model is weaker and while able to learn, its limits are greater and that's probably why L=4 was able to increase accuracy for a longer time and to a better result.
On the other hand when the depth got larger - L=8,16 in the K=32 case and L=16 on the K=64 case, the models weren't able to learn at all, which we explain right below.
2. when L=8 or 16, the models weren't able to learn, this might have been due to the vanishing gradients problem - the more layers there are in the network, the more multiplications are done due to chain rule in order to calculate the gradient, and if some of the values in the way are small, the gradients will vanish and the model parameters will not move. There are some possible solutions for that issue, one can be batch normalization - it will tune the values of the parameters in order to stabilize the learning process and improve the gradient flow. We can also use residual networks, as we learned in class res nets try to solve this issue by creating shortcuts in order to strengthen the signal and avoid gradient vanishing in the backpropagaion process. We will show this solution in 1.4 and compare the results.
"""

part5_q2 = r"""
In experiments 1.2 we managed to train all models except of one (L=8, K=32), all test accuracies were between 60%-70% when many reached above 67.5%. \
We can generally state that higher K has the potential for better test accuracy, we see it in all experiments in part 1.2. \
When L=4 and L=8 we see very similar trends regarding K, we can see that with larger K the model is stronger and able to get higher test accuracy, but can also overfit, this is very clear in the test loss graphs that goes up after reaching good values, slightly after the test accuracy converged, even when using very small early_stopping value (in L=4 we used early_stopping=2) the test loss still went up again before training stopped, when experimenting with higher early_stopping we also saw the test accuracy reducing.
When comparing to experiments 1.1 we can see some similar and some different behaviors - first, regarding K - on both experiments we saw that bigger K can lead to better test accuracies but in experiment 1.1 it wasn't as clear as it is in experiment 1.2. \
Additionally, in 1.2 we used K that wasn't used in 1.1 - K=128 and we see that it is a helpful value that can bring fine results.
"""

part5_q3 = r"""
In experiments 1.3,all three runs achieved a training accuracy of 85-90% and a test accuracy of around 65-70%. Increasing dropout helped with generalization, preventing overfitting and improving test performance. \
However, even with early stopping, the test loss started to increase at some point, indicating that finding the right balance for early stopping is crucial to maintain test performance while avoiding a decrease in test accuracy.\
\\
Different filter sizes, such as [64, 128], provide varying capacities for feature extraction. Using two different K sizes means that the model can capture more varied and complex features, which can enhance its ability to generalize from the training data.
By comaring the plots of 1.1 in L=2,K=64 case and this plot, we can see the complexity introduced by having two different filter sizes per layer can help even shallower models capture more features, leading to higher training accuracy. \
When we tried using a higher weight decay in order to increase the generalization and test performance, models with L=4 layers struggled to learn and vanished. So, while L=4 is not considered very deep, it still requires a balanced approach to regularization.\
\\
With a fixed K, the models achieved approximately the same test accuracy, but training accuracy varied. The model with L=2 layers achieved the highest training accuracy, followed by L=3 and then L=4. 
In addition, deeper models like L=4, while potentially better at capturing complex patterns, face greater challenges during training without skip connections, which can result in lower training accuracy.
\\
In conclusion, increasing the number of layers while keeping the number of convolutional filters constant led to varying training dynamics, with dropout and early stopping playing critical roles in preventing overfitting. The chosen filter sizes provided a balanced capacity for feature extraction, and deeper models required more nuanced tuning of learning rate and regularization to balance effective learning and generalization.

"""

part5_q4 = r"""
Experiment 1.4:\\
In this experiment, we tested the effect of skip connections using a ResNet architecture with six runs: \\

K=[32] fixed with L=8,16,32 \\

K=[64, 128, 256] fixed with L=2,4,8 \\

Results showed that with K=32, the models learned effectively and achieved around 70% test accuracy across different depths.
For K=[64, 128, 256], the depth had a significant impact. The model with L=2 struggled to learn, achieving a 65% test accuracy. 
As the depth increased to L=4 and L=8, test accuracy improved to around 70% and 75%, respectively. Skip connections helped mitigate the vanishing gradient problem, enabling deeper networks to learn more effectively.
\\
Notes about HP tuning and best parameters: \\
- The change in hidden_dims=[512, 256] from [128] was made to enhance the capacity of the fully connected layers at the end of the ResNet convolutional layers. By increasing the hidden dimensions, the network can capture and represent more complex patterns and relationships in the data. This is particularly important when using a varied filter size configuration like [64, 128, 256], as the network processes features at a higher dimention. The larger hidden dimensions allow for better integration and interpretation of these features before the final classification, leading to improved performance.
- The setting pool_every=8 was chosen to prevent the spatial dimensions of the feature maps from becoming too small in deeper architectures. Pooling operations reduce the size of the feature maps, and if performed too frequently, they can shrink the feature maps to a point where they lose valuable information. By spacing the pooling operations every 8 layers, the network maintains larger spatial dimensions, preserving more detailed information throughout the deeper layers of the network. This adjustment is crucial for ensuring that the deeper architectures can still learn effectively from the input data.
- Trying to increase the reg HP decreased the test accuracy for some combination of K-L. So although in some plots, when looking at the training plot it gets to almost perfect training accuracy, becuse we need to fit the hp to all of the expirements in 1.4, we sucrifised some overffiting in several cases for higher test accuracy in others. In an ideal situation we would find the best HP for each combination discribing a different architecture and then find the best architecture for the task in hand.
\\
Comparison of Experiment 1.4 to 1.1 and 1.3\\
Experiment 1.4 vs. Experiment 1.1\\

In Experiment 1.1, deeper networks struggled to learn due to the vanishing gradient problem. As the depth increased, models with fixed K=32 and K=64 showed diminished learning capacity, particularly at L=16. 
In contrast, Experiment 1.4 demonstrated that incorporating skip connections in ResNet architectures significantly improved learning in deeper networks. 
Skip connections mitigated the vanishing gradient problem, enabling models to learn effectively even at greater depths, such as in 1.4 L=8 and L=16 for K=32, and L=4 and L=8 for K=[64, 128, 256].
\\
Experiment 1.3 showed high training accuracy but struggled with deeper networks (L=3, L=4) due to the lack of skip connections. In contrast, Experiment 1.4's skip connections allowed deeper models (L=4, L=8) to perform better, with higher test accuracy and more stable training.
Larger filter sizes in 1.4 (K=[64, 128, 256]) combined with skip connections captured more features and enhanced learning, outperforming the smaller, simpler filters in 1.3. \\
Overall, Experiment 1.4 demonstrated that skip connections and varied filter sizes significantly enhance the performance and generalization of deeper networks, addressing the limitations seen in Experiment 1.1 and 1.3.


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