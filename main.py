from utils import *
from neuronal_network import *
import tensorflow as tf
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import numpy as np
import gym
import time



time.clock=time.time

def main():

    x=input(f'What policy?: \na)epsilon\nb)boltzman\n\nAnswer: ')
    if x=='a':
        x=1
    elif x=='b':
        x=0    


    env = gym.make("LunarLander-v2")
    
    obs_shape=env.observation_space.shape
    n_actions=env.action_space.n

    #Hyperparameters
    time_steps=1
    num_episodes=10000
    lr=0.00025
    lr_decay=1e-6-5e-9
    tao=7
    tao_decay=0.00098
    epsilon=1
    epsilon_decay=0.0002
    b_size=2**11
    batch_size=32
    train_freq=2
    sync_steps=100
    reward_target=490
    e=0.0001
    buffer_beta=0.4
    buffer_alpha=0.6
    beta_augmented=0.00006


    #Buffer and Network instance
    network=DQN(obs_shape,n_actions,lr,tao,epsilon,batch_size)
    buffer=PrioritizedReplayBuffer(b_size,buffer_beta)
    total_rewards=[]
    total_loss=[]
    idx=0
    #network.Q_network.load_weights('C:\ARCHIVOS DE LEO\Inteligencia Artificial\Reinforcement Learning\Codigo\RL springer\Capitulo 4\DQN\CartPole.h5')

    while(len(buffer.storage)!=b_size):
        o=env.reset()
        while(1):
            if x:
                a=network.epsilon_policy(o)
            else:
                a=network.boltzman_policy(o)    
            o_,r,done,info=env.step(a)
            if done:
                d=1
            else:
                d=0
            buffer.add(o,a,r,o_,d)
            print(len(buffer.storage))
            o_x=tf.cast(tf.expand_dims(o,axis=0),dtype='float32')
            a_x=tf.cast(tf.expand_dims(a,axis=0),dtype='int32')
            r_x=tf.cast(tf.expand_dims(r,axis=0),dtype='float32')
            o__x=tf.cast(tf.expand_dims(o_,axis=0),dtype='float32')
            d_x=tf.cast(tf.expand_dims(d,axis=0),dtype='int32')
            priority=tf.abs(network.td_error(o_x,a_x,r_x,o__x,d_x))
            priority=(priority+e)**buffer_alpha
            if idx<b_size:
                buffer.sum_tree[idx]=priority
                buffer.min_tree[idx]=priority
            
            if d:
                break
            idx+=1

    print(buffer.sum_tree._value)       

    for episode in range(num_episodes):

        o=env.reset()
        reward=0
        loss_t=0
        count=0
        while 1:
            if x:
                a=network.epsilon_policy(o)
            else:
                a=network.boltzman_policy(o)    
            o_,r,done,info=env.step(a)
            env.render()
            reward+=r

            if done:
                d=1
            else:
                d=0
            #p=network.get_weights(o,a,r,o_,d)
            buffer.add(o,a,r,o_,d)

            if (time_steps>32 & time_steps%train_freq==0):
                sample_x=buffer.get_sample(batch_size)
                sample=sample_x[0]
                idxs=sample_x[1]
                td_error=tf.abs(network.train(*sample))
                loss_t+=td_error
                priorities=network.get_priorities(td_error,e,buffer_alpha)
                priorities = np.clip(np.abs(priorities), 1e-6, None)
                buffer.sum_tree.update_priorities(idxs,priorities)
                buffer.min_tree.update_priorities(idxs,priorities)

            if time_steps%sync_steps==0:
                network.sync()    
                
            o=o_  
            time_steps += 1


            if done:
                break
            count+=1
        loss_t/=count

        total_rewards.append(reward)
        total_loss.append(loss_t)
        buffer.update_beta(beta_augmented)   

        print(f'EPISODIO: {episode+1} REWARD:{reward} error:{tf.reduce_mean(loss_t)}' )
        if reward>reward_target:
            reward_target=reward
            if x:
                save_model(network.Q_network,'Cart_Pole_PER.h5')
            else:
                save_model(network.Q_network,'Cart_Pole_PER.h5')

        network.update_tao(tao_decay)
        network.update_epsilon(epsilon_decay) 
        #network.update_lr(lr_decay)

    total_rewards=np.array(total_rewards)
    total_loss=np.array(total_loss)
    plt.plot(total_rewards)
    plt.show()
    plt.plot(total_loss)     
    plt.show()


if __name__ == '__main__':
    main()