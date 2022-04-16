# Chap 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
## From rules to data
> In these videos, you were given an introduction to the concepts and paradigms of Machine Learning and Deep Learning. You saw that the traditional paradigm of expressing rules in a coding language may not always work to solve a problem. As such, applications such as computer vision are very difficult to solve with rules-based programming. Instead, if we feed a computer with enough data that we describe (or label) as what we want it to recognize -- given that computers are really good at processing data and matching patterns -- then we could potentially ‘train’ a system to solve a problem. We saw a super simple example of that: fitting numbers to a line. So now, let’s go through a notebook and execute the code that trains a neural network to learn how a set of numbers make up a line. After that, you will feed the trained network a new data point and see if it correctly predicts the expected value.

# Chap 2. Convolutional Neural Networks in TensorFlow
## Using dropout
> The idea behind it is to remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit! 
## Introducing the Rock-Paper-Scissors dataset
> Rock Paper Scissors is a dataset containing 2,892 images of diverse hands in Rock/Paper/Scissors poses. It is licensed CC By 2.0 and available for all purposes, but its intent is primarily for learning and research.

> Rock Paper Scissors contains images from a variety of different hands,  from different races, ages, and genders, posed into Rock / Paper or Scissors and labeled as such. 

> These images have all been generated using CGI techniques as an experiment in determining if a CGI-based dataset can be used for classification against real images. Note that all of this data is posed against a white background.
Each image is 300×300 pixels in 24-bit color

```
From first principles in understanding how ML works, to using a DNN to do basic computer vision, and then beyond into convolutions.

With convolutions, you then saw how to extract features from an image, and you saw the tools in TensorFlow and Keras to build with convolutions and pooling as well as handling complex, multi-sized images.

Through this, you saw how overfitting can have an impact on your classifiers, and explored some strategies to avoid it, including image augmentation, dropout, transfer learning, and more. To wrap things up, you looked at the considerations in your code to build a model for multi-class classification! 
```
# Chap 3. Natural Language Processing in TensorFlow
[News headlines dataset for sarcasm detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)

[Subwords text encoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder)

> Here are the key takeaways for this week:
>> You looked at taking your tokenized words and passing them to an Embedding layer.

>> Embeddings map your vocabulary to vectors in higher-dimensional space. 

>> The semantics of the words were learned when those words were labeled with similar meanings. For example, when looking at movie reviews, those movies with positive sentiment had the dimensionality of their words ending up pointing a particular way, and those with negative sentiment pointing in a different direction. From these, the words in future reviews could have their direction established and your model can infer the sentiment from it. 

>> You then looked at subword tokenization and saw that not only do the meanings of the words matter but also the sequence in which they are found. 
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

[Tensorflow Dataset](https://www.tensorflow.org/datasets/catalog/overview)