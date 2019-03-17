import tensorflow as tf
import pandas as pd
import numpy as np
tf.set_random_seed(1)   # set random seed

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 256
n_inputs = 753               # MNIST data input (img shape: 28*28)
n_steps = 20                # time steps
n_hidden_units = 256        # neurons in hidden layer
n_classes = 753              # MNIST classes (0-9 digits)

x_pre=[]
f=open('train_1.csv')
df=pd.read_csv(f)     #读入股票数据
normalize_data=np.array(df)   #获取最高价序列

#normalize_data=data[:,np.newaxis]       #增加维度

f2 = open('test_1.csv')
df2 = pd.read_csv(f2)  # 读入股票数据
normalize_data2=np.array(df2)   #获取最高价序列
#x_pre=data2[:,np.newaxis]       #增加维度


train_x,train_y=[],[]   #训练集
for i in range(len(normalize_data)-n_steps-1):
    x=normalize_data[i:i+n_steps]
    y=normalize_data[i+n_steps]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

for i in range(len(normalize_data2)-n_steps-1):
    x1=normalize_data2[i:i+n_steps]
    x_pre.append(x1.tolist())


print(len(normalize_data)-n_steps-1)
print(np.array(train_x).shape)
print(np.array(train_y).shape)
print(np.array(x_pre).shape)


# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None,n_classes])


# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(batch):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(x, [-1, n_inputs])

    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell.
    with tf.variable_scope('cell_def', reuse=tf.AUTO_REUSE):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch, dtype=tf.float32) # 初始化全零 state
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    with tf.variable_scope('rnn_def', reuse=tf.AUTO_REUSE):
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output
    return results

def train_lstm():
    global saver
    pred = RNN(batch_size)
    cost =  -tf.reduce_mean(y * tf.log(pred))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    saver = tf.train.Saver(tf.global_variables())
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()
    for i in range(1):
        with tf.Session() as sess:
            sess.run(init)
            start = 0
            step=0
            end = start + batch_size
            while (end<len(train_x)):
                sess.run([train_op,cost], feed_dict={
                    x: train_x[start:end],
                    y: train_y[start:end],
                     })
                start += batch_size
                end = start + batch_size
                if step % 1000 == 0:
                    print(cost,i)
                    print("保存模型：", saver.save(sess, 'module/stock.model'))
            step += 1
train_lstm()
def prediction():
    predict=[]
    pred = RNN(1)
    for i in range(50):
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint('module/')
            saver.restore(sess, module_file)
            next_seq = sess.run(pred, feed_dict={x:x_pre,})
            predict.append(tf.argmax(next_seq, 1))
    dataframe = pd.DataFrame({'LSTM-Result': predict})
    dataframe.to_csv("LSTM-result1.csv", index=False, sep=',')


prediction()


