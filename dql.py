import tensorflow as tf
import numpy as np
import gym
from gym.wrappers import Monitor
import random
import os
import time

def Policy(action_cnn,state,sess,epsilon, num_actions=4): 
	preds = sess.run(action_cnn.preds,{action_cnn.input:state[np.newaxis,:]})
	p = np.ones(num_actions)*epsilon/num_actions
	greedy_action = np.argmax(preds)
	p[greedy_action] += (1 - epsilon)
	return p
	
	

def DQL(env,sess,action_cnn,target_cnn,num_episodes,epsilons,discount_factor,replay_mem_size,batch_size,C,record,decay_rate,algo,make_video = True):
	
	saver = tf.train.Saver()
	# Load a previous checkpoint if we find one
	ckpt = tf.train.get_checkpoint_state(os.path.dirname('Breakoutcheckpoints/'))
	if ckpt and ckpt.model_checkpoint_path:
		print("Loading Checkpoint")
		saver.restore(sess, ckpt.model_checkpoint_path)
	
	decay_count = tf.train.global_step(sess,action_cnn.global_step)
	print(decay_count)
	#Initializing Replay memory
	D = []
	#preprocess = Preprocess()
	state = env.reset()
	state = preprocess(state,sess)
	state = np.stack([state] * 4, axis=2)
	print("populating replay memory")
	start = time.time()
	for i in range(50000):
		action_probs = Policy(action_cnn,state,sess,epsilons[min(decay_count,decay_rate-1)],env.action_space.n)
		possible_actions = np.arange(env.action_space.n)
		action = np.random.choice(possible_actions, p = action_probs)
		new_state,reward,done, _ = env.step(action)
		
		new_state = preprocess(new_state,sess)
		new_state = np.append(state[:,:,1:],new_state[:,:,np.newaxis],axis = 2)
		D.append((state,action,reward,new_state,done))
		if done:
			state = env.reset()
			state = preprocess(state,sess)
			state = np.stack([state] * 4, axis=2)
		else:
			state = new_state
			
	print("Been There Done That")
	print(time.time() - start)
	
	if make_video:
		env = Monitor(env, directory='Breakoutvideos', video_callable=lambda count: count % record == 0, resume=True)
	
	losses = []
	running_mean = 0        
	for i in range(num_episodes):
		saver.save(sess, 'Breakoutcheckpoints/DQN')
		state = env.reset()
		state = preprocess(state,sess)
		state = np.stack([state] * 4, axis=2)
		print("episode: ",i)
		ep_reward = 0
		loss = None
		j =0
		while True:
			print("\rstep {}".format(j),end = "")
			sys.stdout.flush()
			j+=1

			if (decay_count)%C == 0:
				copy_op = copy_cnn_params(action_cnn,target_cnn)
				sess.run(copy_op)
				print("Copied parameters to target network")

			action_probs = Policy(action_cnn,state,sess,epsilons[min(decay_count,decay_rate-1)],env.action_space.n)
			possible_actions = np.arange(env.action_space.n)
			action = np.random.choice(possible_actions, p = action_probs)
			new_state,reward,done,_ = env.step(action)
			new_state = preprocess(new_state,sess)
			new_state = np.append(state[:,:,1:],new_state[:,:,np.newaxis],axis = 2)

			ep_reward += reward
			if(len(D)==replay_mem_size):
				D.pop(0)
			D.append((state,action,reward,new_state,done))
			batch = random.sample(D,batch_size)
			states, actions, rewards, new_states, dones = zip(*batch)
			states = np.array(states)
			actions = np.array(actions)
			rewards = np.array(rewards)
			new_states = np.array(new_states)
			rewards = np.array(rewards)
			flags = np.invert(dones)
			
			#Q Learning
			if algo == 'DQ':
				y = rewards + flags*discount_factor*np.max(sess.run(target_cnn.preds,
																{target_cnn.input:new_states}),axis=1)
			#Double Q Learning
			elif algo == 'DDQ':
				greedy_actions = np.argmax(sess.run(action_cnn.preds,{action_cnn.input:new_states}),axis = 1)
				y = rewards + flags*discount_factor*((sess.run(target_cnn.preds,{target_cnn.input:new_states}))[np.arange(BATCH_SIZE),greedy_actions])

			loss,_ = sess.run([action_cnn.loss,action_cnn.opti],{action_cnn.input:states,action_cnn.output:y,
																 action_cnn.actions:actions})            
			
	 
			state = new_state
			decay_count += 1
			if done:
				break    
		print("Reward: ",ep_reward)
		running_mean = 0.9*running_mean + 0.1*ep_reward
		print("Loss: ",loss)