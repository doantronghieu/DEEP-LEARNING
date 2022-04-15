# Chap 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
## From rules to data
> In these videos, you were given an introduction to the concepts and paradigms of Machine Learning and Deep Learning. You saw that the traditional paradigm of expressing rules in a coding language may not always work to solve a problem. As such, applications such as computer vision are very difficult to solve with rules-based programming. Instead, if we feed a computer with enough data that we describe (or label) as what we want it to recognize -- given that computers are really good at processing data and matching patterns -- then we could potentially ‘train’ a system to solve a problem. We saw a super simple example of that: fitting numbers to a line. So now, let’s go through a notebook and execute the code that trains a neural network to learn how a set of numbers make up a line. After that, you will feed the trained network a new data point and see if it correctly predicts the expected value.

# Chap 2. Convolutional Neural Networks in TensorFlow
## Using dropout
The idea behind it is to remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit! 
# Chap 3. Natural Language Processing in TensorFlow

# Chap 4. Sequences, Time Series and Prediction


# Resources
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

[Bias and techniques to avoid it](https://ai.google/responsibilities/responsible-ai-practices/)

[Image Filtering](https://lodev.org/cgtutor/filtering.html)

[Kernel (image processing)](https://en.wikipedia.org/wiki/Kernel_(image_processing))

[Losses](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

[Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#RMSProp)

[Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)

[Kaggle Dogs v Cats dataset](https://www.kaggle.com/c/dogs-vs-cats)

[Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)