import pandas as pd
import numpy as np
import tensorflow as tf

#——————————————————导入数据——————————————————————
x_pre=[]
train_x,train_y=[],[]   #训练集
f=open('Label-1-1.csv')
df=pd.read_csv(f)     #读入股票数据
data=np.array(df['label'])   #获取最高价序列
normalize_data=data[:,np.newaxis]       #增加维度


f = open('Label-1-3.csv')
df2 = pd.read_csv(f)  # 读入股票数据
data2=np.array(df2['label-2'])   #获取最高价序列
normalize_data2=data2[:,np.newaxis]       #增加维度





#生成训练集
#设置常量
time_step=4      #时间步
rnn_unit=256       #hidden layer units
batch_size=128     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=753    #输出层维度
lr=0.0001         #学习率
keep_prob=0.8

for i in range(len(normalize_data)-time_step+1):
    x = normalize_data[i:i+time_step]
    train_x.append(x.tolist())

for i in range(len(data)-time_step+1):
    y = data[i+1:i+time_step+1]
    train_y.append(y.tolist())


print(len(normalize_data2))

for i in range(len(normalize_data2)-time_step + 1):
    x2=normalize_data2[i:i+time_step]
    x_pre.append(x2.tolist())
print(len(x_pre))


#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.int32, [batch_size,time_step])   #每批次tensor对应的标签
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
        }



#——————————————————定义神经网络变量——————————————————
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell =tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1, output_keep_prob=keep_prob)
    init_state = cell.zero_state(batch, dtype=tf.float32)
    with tf.variable_scope('rnn_def', reuse=tf.AUTO_REUSE):
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state,dtype=tf.float32)
    outputs = tf.unstack(tf.transpose(output_rnn, [1, 0, 2]))
    results = tf.nn.softmax(tf.matmul(outputs[-1], weights['out']) + biases['out'])
    output_rnn = tf.reshape(output_rnn,[-1,rnn_unit])
    output_rnn=tf.nn.softmax(tf.matmul(output_rnn, weights['out']) + biases['out'])
    return output_rnn,results



#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    pred = tf.reshape(pred, [batch_size,time_step,output_size])
    #损失函数
    loss=tf.contrib.seq2seq.sequence_loss(pred,Y,tf.ones([batch_size,time_step], dtype=tf.float32))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #重复训练10000次
        for i in range(5):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #每10步保存一次参数
                if step%500==0 or step==((len(train_x)//batch_size)-1):
                    print(i,step,loss_)
                    print("保存模型：",saver.save(sess,'module3/stock.model'))
                step+=1


train_lstm()


#————————————————预测模型————————————————————
def prediction():
    _,pred1 = lstm(1)  # 预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('module3/')
        saver.restore(sess, module_file)
        predict = []
        # 得到之后100个预测结果
        for i in range(len(x_pre)):
            seq = sess.run(pred1, feed_dict={X: [x_pre[i]]})
            seq = seq[0].tolist()
            A = []
            for j in range(4):
                a = seq.index(max(seq))
                A.append(a)
                seq[a] = 0
            if i % 1000 == 0:
                print('ok:', i)
            predict.append(A)
            np.savetxt('LSTM-ceshi-1.csv', predict, delimiter=',')

prediction()