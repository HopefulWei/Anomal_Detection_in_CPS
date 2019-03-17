import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#——————————————————导入数据——————————————————————
x_pre=[]
f=open('Label-1-4.csv')
df=pd.read_csv(f)     #读入股票数据
data=np.array(df['label'])   #获取最高价序列
normalize_data=data[:,np.newaxis]       #增加维度

f = open('Label-1-3.csv')
df2 = pd.read_csv(f)  # 读入股票数据
data2=np.array(df2['label-2'])   #获取最高价序列
normalize_data2=data2[:,np.newaxis]       #增加维度



#x_pre.append(normalize_data2.tolist())




#生成训练集
#设置常量
time_step=20      #时间步
rnn_unit=256       #hidden layer units
batch_size=128     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=753     #输出层维度
lr=0.001         #学习率
train_x,train_y=[],[]   #训练集
for i in range(len(normalize_data)-time_step-1):
    tf.reset_default_graph()
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+time_step]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
print(len(normalize_data2)-time_step-1)
for i in range(len(normalize_data2)-time_step-1):
    tf.reset_default_graph()
    x=normalize_data2[i:i+time_step]
    x_pre.append(x.tolist())


#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,output_size])   #每批次tensor对应的标签
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }



#——————————————————定义神经网络变量——————————————————
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    with tf.variable_scope('cell_def', reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    with tf.variable_scope('rnn_def', reuse=tf.AUTO_REUSE):
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state,dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    b_out=biases['out']
    pred=tf.matmul(output,output)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    #损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred[-1], [-1]) - tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #重复训练10000次
        for i in range(1):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #每10步保存一次参数
                if step%1000==0:
                    print(i,step,loss_)
                    print("保存模型：",saver.save(sess,'module3/stock.model'))
                step+=1


train_lstm()


#————————————————预测模型————————————————————
def prediction():
    pred, _ = lstm(1)  # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('module3/')
        saver.restore(sess, module_file)
        predict = []
        # 得到之后100个预测结果
        for i in range(2):
            next_seq = sess.run(pred, feed_dict={X: [x_pre[i]]})
            if i %1000==0:
                print('ok:',i)
            next_seq_1=next_seq[19][0]
            predict.append(next_seq_1)
        dataframe = pd.DataFrame({'LSTM Result-1': predict})
        dataframe.to_csv("LSTM-result-1.csv", index=False, sep=',')


prediction()