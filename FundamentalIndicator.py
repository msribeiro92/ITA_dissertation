import numpy as np
import pandas as pd
import tensorflow as tf

from FundamentalData import FundamentalData

class FundamentalIndicator:

    def __init__(
        self,
    ):
        # Env Parameters
        self.fundamentalData = FundamentalData()
        self.n_input = 16
        self.n_classes = 2

    def multilayer_perceptron(
        self,
        X,
        shape=[49,256,256,7]
    ):
        # Store layers weight &amp; bias
        self.weights = {}
        self.biases = {}
        for i in range(1,len(shape)):
            if i < len(shape)-1:
                self.weights['h'+str(i)] = tf.Variable(tf.random_normal([shape[i-1], shape[i]]))
                self.biases['b'+str(i)] = tf.Variable(tf.random_normal([shape[i]]))
            else:
                self.weights['out'] = tf.Variable(tf.random_normal([shape[i-1], shape[i]]))
                self.biases['out'] = tf.Variable(tf.random_normal([shape[i]]))
        # Hidden layer with ReLU activation
        layers = [X]
        for i in range(1,len(shape)-1):
            layer = tf.add(tf.matmul(layers[-1], self.weights['h'+str(i)]), self.biases['b'+str(i)])
            layer = tf.nn.relu(layer)
            layers.append(layer)
        # Output layer with sigmoid activation
        out_layer = tf.add(tf.matmul(layers[-1], self.weights['out']), self.biases['out'])

        return out_layer

    def train(
        self,
        # Set learning hyper parameters
        network_shape=[16,256,256,2],
        learning_rate=0.01,
        batch_size=44,
        num_epochs=300,
    ):
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

        # Construct model
        self.logits = self.multilayer_perceptron(self.X, shape=network_shape)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        # Initializing the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(num_epochs):
                avg_cost = 0.
                total_batch = 6
                #Loop over all batches
                for i in range(total_batch):
                    batch_x = self.fundamentalData.X[i*44:(i+1)*44]
                    #print batch_x[0].shape
                    batch_y = self.fundamentalData.Y[i*44:(i+1)*44]
                    #print batch_y[0].shape
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={self.X: batch_x,
                                                                    self.Y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % 25 == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("Optimization Finished!")
            save_path = saver.save(sess, "/tmp/model.ckpt")

    def dev(
        self,
        shape=[49,256,256,7],
        random_iterations=50,
    ):
        paramSet = []
        for i in range(random_iterations):
            #paratemers to train
            hidden_units = pow(2,int(round(2*np.random.rand()+7)))
            hidden_layers = int(round(2*np.random.rand()+2))
            network_shape = [16]
            for i in range(hidden_layers):
                network_shape.append(hidden_units)
            network_shape.append(2)

            learning_rate = pow(10,-5*np.random.rand()-2)

            paramSet.append({
                'network_shape': network_shape,
                'learning_rate': learning_rate,
            })

        results = []
        for params in paramSet:
            self.train(**params)
            results.append(self.test(is_dev=True))

        return paramSet[results.index(max(results))]

    def test(
        self,
        is_dev=False,
    ):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "/tmp/model.ckpt")

            # Test model
            pred = tf.nn.softmax(self.logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            if is_dev:
                test_index = 6
            else:
                test_index = 8
            result = accuracy.eval({
                self.X: self.fundamentalData.X[test_index*44:(test_index+2)*44],
                self.Y: self.fundamentalData.Y[test_index*44:(test_index+2)*44]
            })

        if not is_dev:
            print "Accuracy: ", result

        return result

    def testSelected(
        self,
    ):
        params = self.dev()
        print 'training with: ' + str(params)
        self.train(**params)
        self.test()
