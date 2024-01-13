# Deep-learning-implementation

This is a deep learning framework (sort of) implementation in Python.

I've used numpy as it's very useful for large numerical computations, like generation of Identity Matrices, Multiplications of Matrices or vectors.

I've also used pyplot from matplotlib to plot graphs of the dataset and the activation values at the output layer.

And to test the model I've used the MNIST's handwritten dataset.

### Test model's architecture

Model's architecture is pretty simple, but it's not the focus. However, for informational purpose, I tested the sort-of-deep-learning framework using following architecture:

    1. Flattener                 : (28, 28) to 784

    2. Weights layer             : 784 to 512
    3. Biases layer              : 512
    4. ReLU Activation

    5. Weights layer             : 512 to 200
    6. Biases layer              : 200
    7. ReLU Activation

    8. Weights layer             : 200 to 50
    9. Biases layer              : 50
    10. ReLU Activation

    11. Weights layer             : 50 to 10
    12. Biases layer              : 10
    13. Softmax Activation, Activated:False

    Total parameters = 515080
    Loss function = Cross Entropy Loss Function from the logits

After training on 140 epochs, each epochs consisting of 1000-train datasets, the model's performance can be seen as the image below:

![Moder performance](Output.png)

## Part of the implementation of DL

This DL implementation is composed of following parts:

1. Layers: Each layer takes a certain input and gives certain output. Each layer is made as simple as possible, and made updatable later.
2. Loss Function: Loss functions takes the output values and the groudn truth values and calculate the loss for a particular case.
3. Model: By far the most complicated part. It consists of all the layers and loss functions, and it must be able to give output given the input, and be trainable.

The real focus is about the implementation of the back-propagation, and how gradients of the outputs are used to calculate the gradients of the parameters and the inputs.

### Making the layers
Making layers is pretty simple, once the mathematics are very clear. One peculiar thing about the layers is, one single example is taken as a row-vector. Thus, all the matrices that are premultiplied, must be transposed and postmultiplied. All the bias layers are also row-vectors.

##### Layers
- Each layers maintains following associated datatypes:
    - Information types
        - name: Name of the layer
        - parameters_count: Number of parameters of the layer
    - Useful types
        - Depends on the type of layer:
            - Eg: For weights layer, it's the weight matrix
            - Eg: For bias layer, it's the bias vector
            - Eg: For the activation layer, it is nothing
        - They can be abstracted away for the user of the layer

- Each layer should have the following functions
    - Apply (input) -> output
        - It applies the layer's operation to it's input to give an output
    - Input gradient (output gradient, input value, output value) -> gradient values for input
        - It calculates the gradient for the input given the gradient values for the output.
        - It takes the output gradient values, it's input values and it's output values, if it saves the computations
    - Parameter gradient (output gradient, input value, output value) -> gradient values for parameters
        - It calculates the gradient for the parameters using the output gradients, input values and output values
        - If the layer doesn't have any parameters, it has to output zero (because we will sum the gradients, and we need something to sum it)
    - dummy gradient () -> a zero of shape of parameters
        - a zero gradient value of shape of the parameters
    - learn (parameter gradient, learning rate)
        - It modifies the parameter given the gradient of the parameters and the learning rate

### Making the Loss Function
Loss function just take the prediction values and ground truth to get a loss value for a particular example in this implementation. For total loss, individual losses are added.

##### Loss function
- Each loss function has following data types
    - Name: Name of the loss function.

- Each loss function should implement the following methods
    - Apply(prediction, ground_truth) -> loss value
        - It calculates the required loss function.
    - Call(prediction, gournd_truth) -> loss value
        - It calls the apply function under the hood.
    - Prediction gradient(prediction, ground_truth) -> gradient values for output
        - It returns the gradient value for the loss wrt prediction value.
        - To calculate it, it is supplied with the prediction value and the ground truth.

### Making the model
It is the most complicated part of the DL implementation.

It takes the input, and using all the layers, forward propagates them until it gets the outputs. Calling the model should do this task.

It's complexity is evident at the backpropagation. First it calculates the gradient for the loss function according to the output, and then it starts to use that gradient values, to back-propagate them back to previous layers. Gradients of the parameters, and the inputs are calculated separately.

Also using this model is quite complicated. But, these are the steps that could be followed:

1. Specify the learning rate.
2. For each epoch do the following things:
    1. Reset the gradients: It resets the gradients value, as well as the loss to zero.
    2. For each train-example do the following:
        1. See the train dataset: (It does following tasks)
            - It forward propagates the training example, saving the intermediate outputs.
            - It calculates the output gradients for the loss. 
            - It uses the output gradients to back-propagates through individual layers to get the parameter gradients as well as input gradient for each layer.
            - It adds the parameter gradients for each layer to the previous sum of gradients.
            - It also adds the loss value.
    3. After seeing many datasets, and remembering the gradients, update the parameters for each layer using the learning rate (the process of learning)


##### Sequential model
- Each sequential model should have following data types
    - Information types
        - Name: Name of the model
    - Useful types
        - Layers: A list of the sequential layers
        - Loss function : The function that defines the loss
        - Dataset transformation : The transformation that is applied to the output of the dataset

        - Cumulative parameter gradients for all layers
        - 
- Each sequential model should implement the following methods
    - Informational / debug
        - representation
    - Useful
        - inference (input) -> output
            - For simple inference
        - Reset gradient
            - To reset the parameter gradients for all layers
        - forward propagation (input) -> output
            - For simple layer by layer forward propagation
            - Saves the input to each layer
            - Saves the last output
            - Prepares for the back propagation
        - Back propagation (output gradients) ->
            - Layer by layer back propagation calculating the parameter gradients at each step, and back-propagating the intermediate gradients
            - Sums the parameter gradients
        - input gradients
            - To calculate the gradient of the input given the gradient of the output

### Important things learned:

##### Learning rate

It's really better if the learning rate adapts to the number of test case it sees. In normal implementation, the loss function takes cares of the number of cases it sees, but for my case I just sum them, and gradients can explode the parameters update if there are too many datasets. Thus, by scaling the learning rate down properly, I get the stable training performance for varying number of test cases seen.

### Future works

First, I added the scaling layer, but haven't tested it. Scaling the input to [0, 1] really helped to train faster, when I used the other frameworks to train the MNIST.

Second, I could make the model more adaptable to the list of inputs, rather than a single training example.

Third, the loss gradients are working properly, but not the loss function. We don't need the loss functions in training process, but we do need them to see how the models are performing as the training progresses. Stabilizing the loss function would help the situation.

### Conclusion

So, yeah, the deep learning model is pretty complicated. But it is doable, and a lot of things can be learned along the way.

By implementing this, I felt so much appreciation for the people that developed the present deep-learning frameworks like TensorFlow, PyTorch, SciKit Learn, etc. They are making job so easy. So, implementing such things myself really shows the things that go in to making them.

Although it's basic, and accuracy is pretty low (can be improved by more training), it works as expected, and gives the correct predictions, and even when they are wrong, they near the correct ones. That is very hopeful.

It was an amazing experience. 