## Model architecture

    Simple MNIST handwritten digit dataset classifier

    1. Scaler with scale factor 0.00392156862745098
    2. Flattener                 : (28, 28) to 784

    3. Weights layer             : 784 to 256
    4. Biases layer              : 256
    5. ReLU Activation

    6. Weights layer             : 256 to 64
    7. Biases layer              : 64
    8. ReLU Activation

    9. Weights layer             : 64 to 10
    10. Biases layer              : 10
    11. Softmax Activation, Activated:False

    Total parameters = 218058
    Loss function = Cross Entropy Loss Function from the logits

## Model's performance

![Model's output](Output.png)


## Model's loss graph

![Model's loss graph](Loss-graph.png)
