import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from ggplot import *
from sklearn.manifold import TSNE
import pandas as pd


def preprocess(state,sess,show = False):
    if len(state.shape) == 4:
        state = state[0]
    img = Image.fromarray(state.astype('uint8'), 'RGB')
    img = img.convert('LA')
    img = img.crop((0,34,160,192))
    img = img.resize((84,84))
    img = np.array(img)
    img = img[:,:,0]
    return img      



def PCA_visualization(frame,feat_cols):

    pca = PCA(n_components = 3)

    pca_fit = pca.fit_transform(frame[feat_cols].values)
    frame['X1'] = pca_fit[:,0]
    frame['X2'] = pca_fit[:,1]

    print(pca.explained_variance_ratio_)

    plot = ggplot(frame, aes(x='X1', y='X2', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
    
    print(plot)
    plot.save('pca.png')


def tSNE_visualization(frame,feat_cols):

    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 400)
    tsne_fit = tsne.fit_transform(frame[feat_cols].values)

    print("TSNE Completed")

    frame['x_tsne'] = tsne_fit[:,0]
    frame['y_tsne'] = tsne_fit[:,1]


    plot = ggplot( frame, aes(x='x_tsne', y='y_tsne', color='label') ) \
            + geom_point(size=70,alpha=0.1) \
            + ggtitle("tSNE dimensions colored by digit")
    print(plot)
    plot.save('tsne.png')


def create_dataframe(env,sess,action_cnn,checkpoint,memsize=5000,epsilon=0.1):

	saver = tf.train.Saver()
	# Load a previous checkpoint if we find one
	ckpt = tf.train.get_checkpoint_state(os.path.dirname('Breakoutcheckpoints/'))
	if ckpt and ckpt.model_checkpoint_path:
		print("Loading Checkpoint")
		saver.restore(sess, ckpt.model_checkpoint_path)

	preprocess = Preprocess()
	state = env.reset()
	state = process(state,sess)
	state = np.stack([state] * 4, axis=2)
	print("Creating Data frame")
	X = []
	y = []
	start = time.time()
	for i in range(memsize):
		action_probs = Policy(action_cnn,state,sess,0.1,env.action_space.n)
		action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
		new_state,reward,done,_ = env.step(action)
		new_state = process(new_state,sess)
		new_state = np.append(state[:,:,1:],np.expand_dims(new_state, 2),axis = 2)
		vislayer = sess.run(action_cnn.vislayer,{action_cnn.input: state[np.newaxis,:]})
		X.append(vislayer[0])
		y.append(action)
		if done:
			state = env.reset()
			state = process(state,sess)
			state = np.stack([state] * 4, axis=2)
		else:
			state = new_state

	X = np.array(X)
	y = np.array(y)

	feat_cols = ['feat'+str(i) for i in range(X.shape[1])]
	print(X.shape)
	frame = pd.DataFrame(X,columns = feat_cols)
	frame['label'] = y
	frame['label'] = frame['label'].apply(lambda i: str(i))

	return frame,feat_cols        