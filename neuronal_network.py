import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Input,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam,RMSprop
import numpy as np
import random
from utils import *


    
def QFunc(obs_shape,n_actions):
    input_layer=Input(obs_shape)
    x=Dense(512,activation='relu',kernel_initializer='he_normal')(input_layer)
    x=Dense(256,activation='relu',kernel_initializer='he_normal')(x)
    x=Dense(64,activation='relu',kernel_initializer='he_normal')(x)
    x=Dense(n_actions,activation='linear')(x)
    model=Model(input_layer,x)
    return model

class DQN:

    def __init__(self,obs_shape,n_actions,lr,tao,epsilon,batch_size):

        super(DQN,self).__init__()
        self.n_actions=n_actions #number of actions
        self.batch_size=batch_size
        self.tao=tao
        self.epsilon=epsilon
        self.optimizer=RMSprop(lr=lr,rho=0.95,epsilon=0.01)
        self.lr=lr
        self.Q_network=QFunc(obs_shape,n_actions)
        self.T_network=QFunc(obs_shape,n_actions)
    

    def sync(self):
        weights=self.Q_network.get_weights()
        self.T_network.set_weights(weights) 

    def boltzman_policy(self,state):

        state=tf.expand_dims(state,axis=0)
        values=self.Q_network(state)[0]
        values=tf.divide(values,tf.constant(self.tao,dtype=tf.float32))
        values=tf.nn.softmax(values).numpy()
        action=random.choices(list(range(self.n_actions)),weights=values,k=1)
        return action[0]

    def epsilon_policy(self,state):
        state=tf.expand_dims(state,axis=0)        
        r=np.random.sample()

        if r<self.epsilon:
            return random.choices(list(range(self.n_actions)),k=1)[0]
        else:
            return self.Q_network(state).numpy().argmax()    

    def update_tao(self,tao_decay):
        
        if self.tao>0.01:
            self.tao-=tao_decay

    def update_epsilon(self,eps_decay):
        if self.epsilon>0.1:
            self.epsilon-=eps_decay        

    def update_lr(self,lr_decay):
        if self.lr>1e-4:
            self.lr-=lr_decay
        self.optimizer=Adam(lr=self.lr)        

    def td_error(self,b_o,b_a,b_r,b_o_,b_d):
        gamma=0.95
        b_a_ = tf.one_hot(tf.argmax(self.Q_network(b_o_), 1), self.n_actions)
        b_q_=tf.cast((1-b_d),tf.float32)*tf.reduce_sum(self.T_network(b_o_)*b_a_,1)
        b_q = tf.reduce_sum(self.Q_network(b_o) * tf.reshape(tf.one_hot(b_a, self.n_actions),(-1,self.n_actions)), 1)
        yi=b_r+gamma*b_q_
        return yi-b_q


    @tf.function
    def train(self,b_o,b_a,b_r,b_o_,b_d,b_w):

        with tf.GradientTape() as tape:
            td_error=self.td_error(b_o,b_a,b_r,b_o_,b_d)
            loss=tf.reduce_mean(self.huber_loss(td_error,1)*b_w)
        grads=tape.gradient(loss,self.Q_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.Q_network.trainable_weights))
        return td_error
    
    def get_priorities(self,td_error,e,alpha):
        priorities=(td_error+e)**alpha
        return priorities

    def huber_loss(self,x,delta):
        return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, delta*tf.abs(x) - delta/2)             

