
# coding: utf-8

# In[1]:


import numpy as np
import gym
from gym.wrappers import Monitor
#For using environments of atari games
import tensorflow as tf
#For building CNNs
import random
import itertools
import os
import sys
import time
import resource
from collections import deque, namedtuple
from sklearn.decomposition import PCA
from ggplot import *
from sklearn.manifold import TSNE
import pandas as pd
# In[2]:


BATCH_SIZE = 32



env = gym.envs.make("Breakout-v0")


#Change
class Preprocess():
    def __init__(self):
        self.placeholder = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.processed = tf.image.rgb_to_grayscale(self.placeholder)
        self.processed = tf.image.crop_to_bounding_box(self.processed, 34, 0, 160, 160)
        self.processed = tf.image.resize_images(
                    self.processed, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.processed = tf.squeeze(self.processed)

    def process(self,state,sess,show = False):
        img = sess.run(self.processed, { self.placeholder: state })

        return img




class DQN():
    def __init__(self,batch_size,scope):
        self.batch_size = batch_size
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.model()
    
    def model(self):
        #tf.reset_default_graph()
            #Placeholders for input(state), output(Value function) and actions
        self.input = tf.placeholder(tf.uint8,shape=[None,84,84,4])
        self.output = tf.placeholder(tf.float32,shape=[None])
        self.actions = tf.placeholder(tf.int32,shape = [None])
            
            #Convolutional Layers
        layers = {}
        initializer = tf.contrib.layers.xavier_initializer()
        X = tf.to_float(self.input) / 255.0
        layers['a1'] = tf.layers.conv2d(X,filters = 32,kernel_size=8,strides=4,
                        padding = 'SAME',kernel_initializer=initializer, name='conv1')
        layers['a1'] = tf.nn.relu(layers['a1'],name='relu1')
        layers['a2'] = tf.layers.conv2d(layers['a1'], filters= 64, kernel_size = 4, strides = 2,
                                                 padding = 'SAME',kernel_initializer=initializer,name='conv2')
        layers['a2'] = tf.nn.relu(layers['a2'],name='relu2')
        layers['a3'] = tf.layers.conv2d(layers['a2'],filters=64,kernel_size=3, strides=1,
                                                 padding = 'SAME',kernel_initializer=initializer, name='conv3')
        layers['a3'] = tf.nn.relu(layers['a3'],name='relu3')
            
            #Fully connected Layers
        layers['a3'] = tf.contrib.layers.flatten(layers['a3'])
        layers['a4'] = tf.layers.dense(layers['a3'],512,kernel_initializer=initializer,name='fc1')
        self.vislayer = layers['a4']
        layers['a4'] = tf.nn.relu(layers['a4'],name='relu4')
        self.preds = tf.layers.dense(layers['a4'],4,kernel_initializer=initializer,name='output')
            
            #Selecting values corresponding to the actions
            #inds = tf.concat([tf.reshape(tf.range(self.batch_size),[1,self.batch_size]),tf.reshape(self.actions,[1,self.batch_size])],axis=0)
            #inds = tf.transpose(inds)
            #self.value_funcs = tf.gather_nd(self.preds,inds)
            
        inds = tf.range(self.batch_size) * tf.shape(self.preds)[1] + self.actions
        self.value_funcs = tf.gather(tf.reshape(self.preds, [-1]), inds)

            
            #Finding the loss
        self.loss = tf.reduce_mean((self.output - self.value_funcs)**2)
            
            #Global Step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
            
            #Optimizer
        self.opti = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss,self.global_step)
            
            
        


# In[8]:


def copy_cnn_params(cnn1,cnn2):
    params_cnn1 = []
    params_cnn2 = []
    for var in tf.trainable_variables():
        if var.name.startswith(cnn1.scope):
            params_cnn1.append(var)
        elif var.name.startswith(cnn2.scope):
            params_cnn2.append(var)
    copy_op = []
    for i in range(len(params_cnn1)):
        copy_op.append(params_cnn2[i].assign(params_cnn1[i]))
    
    return copy_op
    
    

def Policy(action_cnn, num_actions=4):
    
    def Policy_func(state,sess,epsilon):
        preds = sess.run(action_cnn.preds,{action_cnn.input:state[np.newaxis,:]})
        p = np.ones(num_actions)*epsilon/num_actions
        greedy_action = np.argmax(preds)
        p[greedy_action] += (1 - epsilon)
        return p
    
    return Policy_func
    


# In[11]:


def DQL(env,sess,action_cnn,target_cnn,num_episodes,epsilons,discount_factor,replay_mem_size,batch_size,C,record,decay_rate):
    
    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading Checkpoint")
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    decay_count = tf.train.global_step(sess,action_cnn.global_step)
    print(decay_count)
    #Initializing Replay memory
    D = []
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    policy_func = Policy(action_cnn)
    preprocess = Preprocess()
    state = env.reset()
    state = preprocess.process(state,sess)
    state = np.stack([state] * 4, axis=2)
    print("populating replay memory")
    start = time.time()
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    for i in range(50000):
        action_probs = policy_func(state,sess,epsilons[min(decay_count,decay_rate-1)])
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
        new_state,reward,done,_ = env.step(action)
        new_state = preprocess.process(new_state,sess)
        new_state = np.append(state[:,:,1:],np.expand_dims(new_state, 2),axis = 2)
        D.append(Transition(state,action,reward,new_state,done))
        if done:
            state = env.reset()
            state = preprocess.process(state,sess)
            state = np.stack([state] * 4, axis=2)
        else:
            state = new_state
            
    print("Been There Done That")
    print(time.time() - start)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    #env = Monitor(env, directory='videos', video_callable=lambda count: count % record == 0, resume=True)
    losses = []        
    for i in range(num_episodes):
        saver.save(sess, 'checkpoints/DQN')
        state = env.reset()
        state = preprocess.process(state,sess)
        state = np.stack([state] * 4, axis=2)
        print("episode: ",i)
        ep_reward = 0
        loss = None
        for j in itertools.count():
            print("\rstep {}".format(j),end = "")
            sys.stdout.flush()
            action_probs = policy_func(state,sess,epsilons[min(decay_count,decay_rate-1)])
            action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
            new_state,reward,done,_ = env.step(action)
            new_state = preprocess.process(new_state,sess)
            new_state = np.append(state[:,:,1:],new_state[:,:,np.newaxis],axis = 2)

            ep_reward += reward
            if(len(D)==replay_mem_size):
                D.pop(0)
            D.append((state,action,reward,new_state,done))
            batch = random.sample(D,batch_size)
            states, actions, rewards, new_states, dones = map(np.array, zip(*batch))
            flags = np.invert(dones)
            #Q Learning
            #y = rewards + flags*discount_factor*np.max(sess.run(target_cnn.preds,
             #                                                   {target_cnn.input:new_states}),axis=1)
               
            
            #Double Q Learning
            greedy_actions = np.argmax(sess.run(action_cnn.preds,{action_cnn.input:new_states}),axis = 1)
            y = rewards + flags*discount_factor*((sess.run(target_cnn.preds,{target_cnn.input:new_states}))[np.arange(BATCH_SIZE),greedy_actions])

            loss,_ = sess.run([action_cnn.loss,action_cnn.opti],{action_cnn.input:states,action_cnn.output:y,
                                                                 action_cnn.actions:actions})            
            

            if (decay_count+1)%C == 0:
                copy_op = copy_cnn_params(action_cnn,target_cnn)
                sess.run(copy_op)
                print("Copied parameters to target network")
        
            state = new_state
            decay_count += 1
            if done:
                break    
        print("Reward: ",ep_reward)
        print("Loss: ",loss)
        


tf.reset_default_graph()
action_cnn = DQN(BATCH_SIZE,'action_cnn')
target_cnn = DQN(BATCH_SIZE,'target_cnn')
decay_rate = 500000
epsilons = np.linspace(1,0.1,decay_rate)


# In[14]:



def PCA_visualization(frame):

    pca = PCA(n_components = 3)

    pca_fit = pca.fit_transform(frame[feat_cols].values)
    frame['X1'] = pca_fit[:,0]
    frame['X2'] = pca_fit[:,1]

    print(pca.explained_variance_ratio_)

    plot = ggplot(frame.loc[rndperm[:3000],:], aes(x='X1', y='X2', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
    
    plot


def tSNE_visualization(frame):

    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 400)
    tsne_fit = tsne.fit_transform(frame['feat_cols'].values)

    print("TSNE Completed")

    frame['x_tsne'] = tsne_fit[:,0]
    frame['y_tsne'] = tsne_fit[:,1]


    plot = ggplot( frame, aes(x='x_tsne', y='y_tsne', color='label') ) \
            + geom_point(size=70,alpha=0.1) \
            + ggtitle("tSNE dimensions colored by digit")
    plot


def create_dataframe(env,sess,action_cnn,checkpoint,memsize=5000,epsilons=0.1):

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading Checkpoint")
        saver.restore(sess, ckpt.model_checkpoint_path)

    policy_func = Policy(action_cnn)
    preprocess = Preprocess()
    state = env.reset()
    state = preprocess.process(state,sess)
    state = np.stack([state] * 4, axis=2)
    print("populating replay memory")
    X = []
    y = []
    start = time.time()
    for i in range(memsize):
        action_probs = policy_func(state,sess,epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
        new_state,reward,done,_ = env.step(action)
        new_state = preprocess.process(new_state,sess)
        new_state = np.append(state[:,:,1:],np.expand_dims(new_state, 2),axis = 2)
        vislayer = sess.run(action_cnn.vislayer,{action_cnn.input: state})
        X.append(vislayer)
        y.append(action)
        if done:
            state = env.reset()
            state = preprocess.process(state,sess)
            state = np.stack([state] * 4, axis=2)
        else:
            state = new_state

    X = np.array(X)
    y = np.array(y)

    feat_cols = ['feat'+str(i) for i in range(X.shape[1])]
    frame = pd.DataFrame(X,columns = feat_cols)
    frame['labels'] = y
    frame['label'] = frame['label'].apply(lambda i: str(i))

    return frame        


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    copy_op = copy_cnn_params(action_cnn,target_cnn)
    sess.run(copy_op)
    DQL(env,sess,action_cnn,target_cnn,10000,epsilons,discount_factor=0.99,replay_mem_size=500000,
        batch_size=BATCH_SIZE,C=10000,record=50,decay_rate=decay_rate)
    


# In[20]:


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint('checkpoints/')
    print(checkpoint)
    if checkpoint:
        print("Loading model checkpoint {}...\n".format(checkpoint))
        saver.restore(sess, checkpoint)
    #ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'))
    #if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)
        
    print("Playing")
    state = env1.reset()
    for t in range(50000):
        state = preprocess(state)
        env1.render()
        action_probs = policy_func(state,sess,epsilons[decay_rate-1],4,action_cnn)
        action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
        state,reward,done,_ = env1.step(action)
        if done:
            print("Game Over")
            break
    print("Played")
    
    

