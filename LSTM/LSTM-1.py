import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data  #从TensorFlow在线导入数据集
from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt
import numpy as np
import pylab
config = tf.ConfigProto() #tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置：
sess = tf.Session(config=config) #调用会话，和调用计算机GPU有关
correct_prediction = []


###########################################################输入##########################################################

f=open('label-1.csv')
df=pd.read_csv(f)     #读入股票数据
normalize_data=np.array(df['label'])   #获取最高价序列


f2 = open('label-3.csv')
df2 = pd.read_csv(f2)  # 读入股票数据
data2=np.array(df2['label-2'])   #获取最高价序列
normalize_data2=data2[:,np.newaxis]       #增加维度






############################################################参数#########################################################
lr = 1e-3
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = 128
# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size =1
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size =20
# 每个隐含层的节点数
hidden_size = 5
# LSTM layer 的层数
layer_num = 256
# 最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 1   #753
_X = tf.placeholder(tf.float32, [None, 20])        #此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值，[None, 784]表示数据形状，列是784，行不定
y = tf.placeholder(tf.float32, [None,timestep_size,class_num])#tf.placeholder 用于得到传递进来的真实的训练样本,placeholder，中文意思是占位符，在tensorflow中类似于函数参数，运行时必须传入值。
keep_prob = tf.placeholder(tf.float32)
X = tf.reshape(_X, [-1, 1, 20])


train_x,train_y, pre_x=[],[],[]  #训练集
for i in range(len(normalize_data)-timestep_size-1):
    tf.reset_default_graph()
    x=normalize_data[i:i+timestep_size]
    a=normalize_data[i+timestep_size]
    train_x.append(x.tolist())
    train_y.append(a.tolist())
#print(len(normalize_data2)-timestep_size-1)

for i in range(len(normalize_data2)-timestep_size-1):
    tf.reset_default_graph()
    z=normalize_data2[i]
    pre_x.append(z.tolist())




###############################################定义 LSTM 结构#############################################################
def unit_lstm():
    # 定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    #添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell
#调用 MultiRNNCell 来实现多层 LSTM
mlstm_cell = rnn.MultiRNNCell([unit_lstm() for i in range(3)], state_is_tuple=True)



#############################################初始化#######################################################################
#用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)   #state表示由(c,h)组成的tuple  所以dynamic_rnn实现的功能就是可以让不同迭代传入的batch可以是长度不同数据，但同一次迭代一个batch内部的所有数据长度仍然是固定的
h_state = state[-1][1]  # 或者 h_state = state[-1][1]？？？


################################################设置 loss function 和 优化器##############################################
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)  #主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）；
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(max[y_pre]))
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
prediction=tf.argmax(y_pre,1)
correct_prediction = tf.equal(tf.argmax(y_pre,1),y )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())    #和训练精度有关
for i in range(1):
    _batch_size = 128
    if (i + 1) % 2 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X: train_x, y: train_y, keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("Iter%d, step %d, training accuracy %g" % (train_x.epochs_completed, (i + 1), train_accuracy))

    sess.run(train_op, feed_dict={_X: train_x, y:train_y, keep_prob: 0.5, batch_size: _batch_size})

print("test accuracy %g"% sess.run(accuracy, feed_dict={ _X: train_x, y: train_y, keep_prob: 1.0, batch_size:128}))

def prediction():
    predict=[]
    for i in range(50):
        next_seq = sess.run(prediction, feed_dict={_X: pre_x, keep_prob: 1.0})
        predict.append(next_seq)
    dataframe = pd.DataFrame({'LSTM-Result': predict})
    dataframe.to_csv("LSTM-result1.csv", index=False, sep=',')


#y_top = tf.nn.top_k(y_pre,k=4,sorted=True,name=None)




