from model import *
from dql import *
from prep_vis import *

BATCH_SIZE = 32


def main(option):
	env = gym.envs.make("Breakout-v0")
	tf.reset_default_graph()
	action_cnn = DQN(BATCH_SIZE,'action_cnn',env.action_space.n)
	target_cnn = DQN(BATCH_SIZE,'target_cnn',env.action_space.n)
	decay_rate = 500000
	epsilons = np.linspace(1,0.1,decay_rate)

	if option == 'train':

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			DQL(env,sess,action_cnn,target_cnn,10000,epsilons,discount_factor=0.99,replay_mem_size=500000,
				batch_size=BATCH_SIZE,C=10000,record=50,decay_rate=decay_rate,algo='DDQ',make_video = False)
	
	if option == 'play':		

		with tf.Session() as sess:
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			checkpoint = tf.train.latest_checkpoint('Breakoutcheckpoints/')
			if checkpoint:
				print("Loading model checkpoint {}...\n".format(checkpoint))
				saver.restore(sess, checkpoint)

			else:
				print("No checkpoint found, playing randomly")
				state = env.reset()
				for t in range(500):
					env.render()
					action = env.action_space.sample()
					state,reward,done,_ = env.step(action)
					if done:
						break

				return 0

		
			print("Playing")
			#preprocess = Preprocess()
			state = env.reset()
			state = preprocess(state,sess)
			state = np.stack([state] * 4, axis=2)
			while True:
				env.render()
				action_probs = Policy(action_cnn,state,sess,0.1,env.action_space.n)
				possible_actions = np.arange(env.action_space.n)
				action = np.random.choice(possible_actions, p = action_probs)
				new_state,reward,done, _ = env.step(action)
		
				new_state = preprocess(new_state,sess)
				new_state = np.append(state[:,:,1:],new_state[:,:,np.newaxis],axis = 2)
			
				if done:
					break
			
				state = new_state	

	if option == 'vis':
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			frame,feat_cols = create_dataframe(env,sess,action_cnn,'Breakoutcheckpoints/')
			print("Performing PCA_visualization")
			PCA_visualization(frame,feat_cols)
			print("Performing tSNE_visualization")
			tSNE_visualization(frame,feat_cols)	



if __name__ == '__main__':
	option = input("Play, Train or Visualize")
	main(option)