import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ''  # use GPU with ID=0

#——————————————————导入数据——————————————————————
x_pre=[]
f=open('train1-2.csv')
df=pd.read_csv(f)     #读入股票数据
normalize_data=np.array(df)   #获取最高价序列
i1 = 0
i2 = 0
A = []
f2 = open('test1-2.csv')
df2 = pd.read_csv(f2)  # 读入股票数据
test_data=np.array(df2)   #获取最高价序列

with open('segmentation.csv',"r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[0] == '1':
            A.append(i1)
        i1 = i1 + 1
    A.append(1000000)
#生成训练集
#设置常量
time_step=10       #时间步
rnn_unit=128      #hidden layer units
batch_size=64     #每一批次训练多少个样例
input_size=756     #输入层维度
output_size=756     #输出层维度
lr=0.001         #学习率
train_x,train_y=[],[]   #训练集

for i in range(len(normalize_data)-time_step):
    if i+time_step > A[k]:
        i = A[k] + 1
        k = k + 1
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+time_step]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

for i in range(len(test_data)-time_step+1):
    x1=test_data[i:i+time_step]
    x_pre.append(x1.tolist())


#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,output_size])   #每批次tensor对应的标签
  #每批次tensor对应的标签
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
        }





#——————————
# ————————定义神经网络变量——————————————————
def lstm(batch,reuse=False):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    if reuse:                                        ### 改动部分 ###
        vs.get_variable_scope().reuse_variables()    ### 改动部分 ###
    outputs, states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    outputs= tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.nn.softmax(tf.matmul(outputs[-1], weights['out']) + biases['out'])
    return results



#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred=lstm(batch_size)
    #损失函数
    loss=-tf.reduce_mean(Y * tf.log(pred))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #重复训练10000次
        for i in range(5):
            step=0
            start=0
            losses = 0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                losses = loss_ + losses
                end=start+batch_size
                #每10步保存一次参数
                if step % 500 == 0 or step == ((len(train_x) // batch_size) - 1):
                    print(i,step,loss_,losses)
                    print("保存模型：",saver.save(sess,'module3/stock.model'))
                step+=1


train_lstm()


#————————————————预测模型————————————————————
def prediction():
    pred1= lstm(1,reuse=True)  # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('module3/')
        saver.restore(sess, module_file)
        predict = []
        # 得到之后100个预测结果
        for i in range(len(x_pre)):
            seq = sess.run(pred1, feed_dict={X: [x_pre[i]]})
            seq=seq[0]
            seq= seq.tolist()
            A=[]
            for j in range(4):
                a=seq.index(max(seq))
                A.append(a)
                seq[a]=0
            if i %1000==0:
                print('ok:',i)
            predict.append(A)
            np.savetxt('LSTM-result-1.csv', predict, delimiter=',')



prediction()