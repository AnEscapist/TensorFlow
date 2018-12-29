Reference: https://becominghuman.ai/creating-your-own-neural-network-using-tensorflow-fa8ca7cc4d0e

# Creating you own neural network using TensorFlow

TensorFlow is a great ML/DL library which can be used to implement almost any ML/DL algorithms in a convenient and efficient manner.

* First step:

```python
import tensorflow as tf
```

* Next:

Set the hyper-parameters of your network:
```python
training_epochs = 500
n_neurons_h1 = 60
n_neurons_h2 = 60
learning_rate = 0.01
```

## Training epochs
A epoch is completed when we have used all our training data for the training process. Training data consist of our training features
and it's corresponding training labels. Here we have set training epochs to 500 which means that we train on our entire data on 500 iterations. There is no ideal number of training epochs. Theis depends on the complexity of your data. Therefore, you should do parameter tuning or basically try few parameter configurations to find the ideal/suitable value of this parameters.

Since we care implementing a multi-layer neural network, it will consist of an input layer, two/multiple hidden layers and one output layer.

## Numbers of neurons in the hidden layers

Hiden layers are the layers which perform transforms on the input data to identify patterns and generalize our model. Here I have used 30 neurons each in my first and second hidden layers which was sufficient in achieving a decent accuracy. But as I explained earlier all hyper-parameters should be tuned in such a way that it improves your model. 

## Learning rate

This is the phase at which the algorithm learns. ML gurus say that we should start with a high learning rate and gradually reduce it to achieve the best results. Further the learning rate is advised to be kept within the range of 0 and 1.

## Creating Placeholders

Placeholder is a special type of data handler which facilitates receiving inputs during runtime.

Here we create two placeholders X and Y. Where X holds the input features adn Y holds the corresponding input labels/targets. X and Y are both tensors (tensor is the central unit of data in TensorFlow).

```python
X = tf.placeholder(tf.float32, [None, n_features], name='features')
Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')
```
## Specifying weights and biases for the first layer

Weights allow you to change  the steepness of the activation function in such a way that you will yield better results. While the biases allow you to shift your activation function left or right. Both these parameters are important in most cases.

The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
```python
w1 = tf.Variable(tf.truncated_normal([n_features, n_neurons_h1], mean=0, stddev=1/np.sqrt(n_features)), name='weightes1')
b1 = tf.Variable(tf.truncated_normal([n_neurons_h1], mean=0, stddev=1/np.sqrt(n_features)), name='bias1')
```

## Adding an activation function

There are many popular activation functions among them relu and sigmoid are in front. Relu has the advantage of operating without been affected by the gradient vanishing problem which sigmoid is vulnerable to.

The activation function to the first hidden layer can be added as follows:

```python
y1 = tf.nn.tanh(tf.matmul(X, w1) + b, name='activationLayer1')
```
I’ll also add the weights ,biases and the activation function for the second layer and the output layer as above.

```python
# set parameters for layer 2
w2 = tf.Variable(tf.random_normal([n_neurons_h1, n_neurons_h2], mean=0, stddev=1/np.sqrt(n_features)), name='weights2')
b2 = tf.Variable(tf.random_normal([n_neurons_h2], mean=0, stddev=1/np.sqrt(n_features)), name='bias2')

y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b, name='activationLayer2')
```

```python
# output layer
w0 = tf.Variable(tf.random_normal([n_neurons_h2, n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='weightOut')
b0 = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_features)), name='biasOut')
a = tf.nn.softmax(tf.matmul(y2, w0) + b0, name='activationOutputLayer')
```

## Setting up the cost function and optimizer

Here the cost function is based on the cross entropy calculation where the actual label is multiplied by the log of the predicted label adn teh sum of that is derived. Then reduce mean takes mean along dimensions while dimensions are reduced according to the reduction indices parameter value.

Train step is the optimization function (gradient descent in this scenario) that adjusts weights and biases by a fraction of the learning rate in the direction which results in a cost reduction.

```python
#cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a), reduction_indices=[1]))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cross_entropy)
```
## Accuracy calculation

Here we first match the prediction with the actual afterwards we compute the accuracy by checking the amount of total correct predictions over the total amount of data.

```python
#compare predicted value from network with the expected value/target
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
#accuracy determination
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
```
## Training the model

Now since all are setup we can now train our model by feeding the values in the placeholders. We will be training the model over the number of iterations specified for the training-epochs variable.

```python
# initialization of all variables
initial = tf.global_variables_initializer()

#creating a session
with tf.Session() as sess:
    sess.run(initial)
    writer = tf.summary.FileWriter("/home/tharindra/PycharmProjects/WorkBench/FinalYearProjectBackup/Geetha/TrainResults")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()
    
    # training loop over the number of epoches
    batchsize=10
    for epoch in range(training_epochs):
        for i in range(len(tr_features)):

            start=i
            end=i+batchsize
            x_batch=tr_features[start:end]
            y_batch=tr_labels[start:end]
            
            # feeding training data/examples
            sess.run(train_step, feed_dict={X:x_batch , Y:y_batch,keep_prob:0.5})
            i+=batchsize
        # feeding testing data to determine model accuracy
        y_pred = sess.run(tf.argmax(a, 1), feed_dict={X: ts_features,keep_prob:1.0})
        y_true = sess.run(tf.argmax(ts_labels, 1))
        summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: ts_features, Y: ts_labels,keep_prob:1.0})
        # write results to summary file
        writer.add_summary(summary, epoch)
        # print accuracy for each epoch
        print('epoch',epoch, acc)
        print ('---------------')
        print(y_pred, y_true)
```

# Summary

So basically first we have to feed our data to placeholders.Transform the data via the hidden layers ,output the probabilistic values via the output layer for each label.Calculate the accuracy, minimize the cost throughout the training epochs by adjusting weights and biases.








