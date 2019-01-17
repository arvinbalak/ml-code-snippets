import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load data
data = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.20, stratify=data.target)

# normalize data input
X_scaler = MinMaxScaler(feature_range=(0, 1))
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.fit_transform(X_test)

# one hot encode output
onehot_encoder = OneHotEncoder(sparse=False)
Y_train = onehot_encoder.fit_transform(Y_train.reshape(-1, 1))
Y_test = onehot_encoder.fit_transform(Y_test.reshape(-1, 1))

# parameters and constants
log_interval = 5

num_features = X_train.shape[1]
num_samples = X_train.shape[0]
num_output = Y_train.shape[1]

lr = 0.01
epochs = 10000

hidden1_nodes = 150

# model specs
X = tf.placeholder(tf.float32, (None, num_features))

w_1 = tf.get_variable('weights1', shape=(num_features,hidden1_nodes))
b_1 = tf.get_variable('bias1', shape=(hidden1_nodes))
output1 = tf.nn.relu(tf.matmul(X,w_1)+b_1)

w_2 = tf.get_variable('weights2', shape=(hidden1_nodes,num_output))
b_2 = tf.get_variable('bias2', shape=(num_output))
prediction = tf.matmul(output1,w_2)+b_2

Y = tf.placeholder(tf.float32, (None, num_output))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction))

optimizer = tf.train.AdagradOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(tf.argmax(Y, axis=1), tf.argmax(prediction,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# training summary
tf.summary.scalar('current_cost', cost)
tf.summary.scalar('current_accuracy', accuracy)
summary = tf.summary.merge_all()

# training
with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            session.run(optimizer,feed_dict={X:X_train,Y:Y_train})

            # Every 5 training steps, log our progress
            if epoch % log_interval == 0:
                training_cost, training_summary = session.run([cost, summary],
                                                              feed_dict={X: X_train, Y: Y_train})
                testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: X_test, Y: Y_test})

                # accuracy
                train_accuracy = session.run(accuracy, feed_dict={X: X_train, Y: Y_train})
                test_accuracy = session.run(accuracy, feed_dict={X: X_test, Y: Y_test})

                print(epoch, training_cost, testing_cost, train_accuracy, test_accuracy)

        # Training is now complete!
        print("Training is complete!\n")

        final_train_accuracy = session.run(accuracy, feed_dict={X: X_train, Y: Y_train})
        final_test_accuracy = session.run(accuracy, feed_dict={X: X_test, Y: Y_test})

        print("Final Training Accuracy: {}".format(final_train_accuracy))
        print("Final Testing Accuracy: {}".format(final_test_accuracy))


