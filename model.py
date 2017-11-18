import tensorflow as tf

class DQN():
	def __init__(self,batch_size,scope,output_size):
		self.batch_size = batch_size
		self.scope = scope
		with tf.variable_scope(self.scope):
			self.model(output_size)
	
	def model(self,output_size):
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


		_,a,b,c  = layers['a3'].shape
		shape = a*b*c
		layers['a4'] = tf.reshape(layers['a3'],[-1,shape])
		layers['a4'] = tf.layers.dense(layers['a4'],512,kernel_initializer = initializer,name = 'fc1')
		layers['a4'] = tf.nn.relu(layers['a4'])
		self.preds = tf.layers.dense(layers['a4'],output_size,kernel_initializer = initializer, name = 'output')
		self.vislayer = layers['a4']

			#Selecting values corresponding to the actions
		inds = tf.concat([tf.reshape(tf.range(self.batch_size),[1,self.batch_size]),tf.reshape(self.actions,[1,self.batch_size])],axis=0)
		inds = tf.transpose(inds)
		self.value_funcs = tf.gather_nd(self.preds,inds)

			
			#Finding the loss
		self.loss = tf.reduce_mean((self.output - self.value_funcs)**2)
			
			#Global Step
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
			
			#Optimizer
		self.opti = tf.train.RMSPropOptimizer(learning_rate =0.00025, decay =  0.99, momentum = 0.0, epsilon = 1e-6).minimize(self.loss,self.global_step)



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