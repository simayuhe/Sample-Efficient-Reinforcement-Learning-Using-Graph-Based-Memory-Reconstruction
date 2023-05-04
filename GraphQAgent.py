from __future__ import division

import numpy as np
import tensorflow as tf
import scipy#.misc.imresize
#import cv2
from skimage.transform import resize

import time
from G_Dict import Graph_dict

from collections import Counter

class GraphQAgent():
    def __init__(self, session, args):

        # Environment details
        self.obs_size = list(args.obs_size)
        self.n_actions = args.num_actions
        self.viewer = None

        # Agent parameters
        self.discount = args.discount
        self.n_steps = args.n_step
        self.initial_epsilon = args.epsilon
        self.epsilon = self.initial_epsilon
        self.epsilon_final = args.epsilon_final
        self.epsilon_anneal = args.epsilon_anneal


        # GQ_dict parameters
        self.Graph_size = args.memory_size
        self.delta = args.delta
        self.dict_delta = args.delta#0.1
        self.alpha = args.alpha
        self.number_nn = args.num_neighbours
        self.dist_th = args.dist_th
        # Training parameters
        self.model = args.model
        self.history_len = args.history_len
        self.memory_size = args.replay_memory_size
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.learn_step = args.learn_step

        # Stored variables
        self.step = 0
        self.old_step =0 # 用来给轨迹计数的
        self.started_training = False
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.session = session

        # Replay Memory
        self.memory = ReplayMemory(self.memory_size, self.obs_size)

        # key points memory
        self.keypoint= ExpertMemory(args.expert_memory_size)# 先记10条

        # Preprocessor:
        if args.preprocessor == 'deepmind':
            self.preproc = deepmind_preprocessor
        elif args.preprocessor == 'grayscale':
            #incorrect spelling in order to not confuse those silly americans
            self.preproc = greyscale_preprocessor
        else:
            self.preproc = default_preprocessor
            #a lambda could be used here, but I think this makes more sense
        

        # Tensorflow variables:
        
        # Model for Embeddings
        with tf.variable_scope('agent_model'):
          if self.model == 'CNN':
            from networks import deepmind_CNN
            self.state = tf.placeholder("float", [None, self.history_len]+self.obs_size)
            self.state_embeddings, self.weights = deepmind_CNN(self.state, seed=self.seed)
          elif self.model == 'nn':
            from networks import feedforward_network
            self.state = tf.placeholder("float", [None]+self.obs_size)
            self.state_embeddings, self.weights = \
              feedforward_network(self.state, seed=self.seed)
          elif self.model == 'object':
            from networks import embedding_network
            self.state = tf.placeholder("float", [None]+self.obs_size)
            # mask to enable masking out of entries, last dim is kept for easy broadcasting
            self.masks = tf.placeholder("float", [None, None, 1])
            #tf.Variable(tf.ones("float", [None, None, 1]))
            self.state_embeddings, self.weights = \
              embedding_network(self.state, self.masks, seed=self.seed)

        self.G= Graph_dict(self.Graph_size,self.state_embeddings.get_shape()[-1],self.n_actions,self.dist_th)

        self.action = tf.placeholder(tf.int8, [None])

        self.embs_and_values = tf.py_func(self.G.query_by_features,[self.state_embeddings, self.action, self.number_nn], [tf.float64,tf.float64])
        self.G_values = tf.cast(self.embs_and_values[0],dtype=tf.float32)
        self.G_embeddings = tf.cast(self.embs_and_values[1],dtype=tf.float32)
        square_diff = tf.square(self.G_embeddings - tf.expand_dims(self.state_embeddings, 1))
        distances = tf.reduce_sum(square_diff, axis=2) + [self.delta] # 加和发生在第三个维度
        weightings = 1.0 / distances
        self.normalised_weightings = weightings / tf.reduce_sum(weightings, axis=1, keep_dims=True)
        self.pred_q_G = tf.reduce_sum(self.G_values*self.normalised_weightings, axis=1)
        # Loss Function
        self.target_q = tf.placeholder("float", [None])
        self.td_err_G = self.target_q-self.pred_q_G
        total_loss_G = tf.reduce_sum(tf.square(self.td_err_G))
        # Optimiser

        self.optim_G = tf.train.RMSPropOptimizer(
          self.learning_rate, decay=0.9, epsilon=0.01).minimize(total_loss_G)
        self.model_weights = tf.get_collection(tf.GraphKeys.VARIABLES, scope='agent_model')
        self.saver = tf.train.Saver(self.model_weights)


    def _get_state(self, t=-1):
        # Returns the compiled state from stored observations
        if t==-1: t = self.trajectory_t-1

        if self.history_len == 0:
            state = self.trajectory_observations[t]
        else:
            if self.obs_size[0] == None:
                state = []
                for i in range(self.history_len):
                    state.append(self.trajectory_observations[t-i])
            else:
                state = np.zeros([self.history_len]+self.obs_size)
                for i in range(self.history_len):
                  if (t-i) >= 0:
                    state[i] = self.trajectory_observations[t-i]
        return state


    def _get_state_embeddings(self, states):
        # Returns the DND hashes for the given states
        if self.obs_size[0] == None:
            states_, masks = batch_objects(states)
            embeddings = self.session.run(self.state_embeddings,
              feed_dict={self.state: states_, self.masks: masks})
        else:    
            embeddings = self.session.run(self.state_embeddings, feed_dict={self.state: states})
        return embeddings


    def _predict_g(self, embedding):
        # Return action values for given embedding

        # calculate Q-values
        qs = []
        for a in range(self.n_actions):
            if self.G.queryable(a,self.number_nn):
              q = self.session.run(self.pred_q_G, feed_dict={
                self.state_embeddings: [embedding], self.action: [a]})[0]
            else:
              q = 0.0
            qs.append(q)

        # Return Q values
        return qs


    def _train_g(self, states, actions, Q_targets):
        
        for i in range(self.n_actions):
          if not self.G.queryable(i,self.number_nn):
            return False

        # # print("train_g")
        self.started_training = True

        if self.obs_size[0] == None:
            states_, masks = batch_objects(states)

            feed_dict = {
              self.state: states_,
              self.masks: masks,
              self.target_q: Q_targets,
              self.action: actions
            }

        else:
            feed_dict = {
              self.state: states,
              self.target_q: Q_targets,
              self.action: actions
            }
        self.session.run(self.optim_G, feed_dict=feed_dict)
        
        return True

    def Reset(self, obs, train=True):
        self.training = train

        #TODO: turn these lists into a proper trajectory object
        self.trajectory_observations = [self.preproc(obs)]
        self.trajectory_embeddings = []
        self.trajectory_values = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_t = 0
        self.trajectory_nodth = [] # 用来存储状态在图中的节点标签

        return True

    def GetAction_wq(self,step):
        state = self._get_state()
        embedding = self._get_state_embeddings([state])[0]
        qs_by_g = self._predict_g(embedding)

        if step-self.old_step>0.9*sum(self.keypoint.max_steps)/self.keypoint.memory_size: #self.training:# 先让代码跑一会儿，不进行用记忆进行决策，方便探索更广阔的空间
            #print("step",step,"old step",self.old_step,"max_steps",self.keypoint.max_steps)
            action = self.rng.randint(0, self.n_actions)
        else:
          if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.n_actions)
          else:
            maxq= np.max(qs_by_g)
            a_candidate =[]
            for i in range(self.n_actions):
              if qs_by_g[i]==maxq:
                  a_candidate.append(i)
            action =np.random.choice(a_candidate)
            #value = Qs[action] # Paper uses maxQ, uncomment for on-policy updates # 这里是不是要更改value值呢？
        value = qs_by_g[action]

        self.trajectory_embeddings.append(embedding)
        self.trajectory_values.append(value)
        return action, value
        
    def Update(self, action, reward, obs, terminal=False):

        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)
        self.trajectory_t += 1
        self.trajectory_observations.append(self.preproc(obs))

        self.step += 1

        if self.training:

            # Update Epsilon
            per = min(self.step / self.epsilon_anneal, 1)
            self.epsilon = (1-per)*self.initial_epsilon + per*self.epsilon_final
            #print("--------------memory count",self.memory.count,"batch size ",self.batch_size,"step ",self.step,"lear step",self.learn_step)
            if self.memory.count > self.batch_size*2 and (self.step % self.learn_step) == 0:
                # Get transition sample from memory
                s, a, R = self.memory.sample(self.batch_size, self.history_len)
                # Run optimization op (backprop)
                self._train_g(s, a, R)



            # Add to replay memory and DND
            if terminal:
                temp_steps =self.step-self.old_step
                self.old_step= self.step
                # Calculate returns
                returns = []
                for t in range(self.trajectory_t):
                    if self.trajectory_t - t > self.n_steps:
                        #Get truncated return
                        start_t = t + self.n_steps
                        R_t = self.trajectory_values[start_t]
                    else:
                        start_t = self.trajectory_t
                        R_t = 0
                        
                    for i in range(start_t-1, t, -1):
                        R_t = R_t * self.discount + self.trajectory_rewards[i]
                    returns.append(R_t)
                    self.memory.add(self.trajectory_observations[t], self.trajectory_actions[t], R_t, (t==(self.trajectory_t-1)))
                time_add_a = time.time()
                encoded_trj = self.G.add_by_features(self.trajectory_embeddings,self.trajectory_actions, returns)
                R = sum(self.trajectory_rewards)
                self.keypoint.add_focus(encoded_trj,self.trajectory_rewards,self.trajectory_observations,temp_steps)

                time_add_b = time.time()
                print("time using ",time_add_b-time_add_a)
                


        return True
        
        
    def Save(self, save_dir):
        self.saver.save(self.session, save_dir + '/model.ckpt')
        self.DND.save(save_dir + '/DNDdict')

    def Load(self, save_dir):
        ckpt = tf.train.get_checkpoint_state(save_dir)
        print("Loading model from {}".format(ckpt.model_checkpoint_path))
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        self.DND.load(save_dir + '/DNDdict')


def batch_objects(input_list):
    # Takes an input list of lists (of vectors), pads each list the length of the longest list,
    #   compiles the list into a single n x m x d array, and returns a corresponding n x m x 1 mask.
    max_len = 0
    out = []; masks = []
    for i in input_list: max_len = max(len(i),max_len)
    for l in input_list:
        # Zero pad output
        out.append(np.pad(np.array(l,dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
        # Create mask...
        masks.append(np.pad(np.array(np.ones((len(l),1)),dtype=np.float32), ((0,max_len-len(l)),(0,0)), mode='constant'))
    return out, masks

class ExpertMemory():
  def __init__(self,memory_size=10):
    self.memory_size = memory_size# 最多能记录多少条轨迹
    self.returns = [0]
    self.trjs = [list([0])]
    self.obss = [[np.zeros([2,2])]]
    self.max_steps =[0]
    


  def add_focus(self,nodes,rewards,obses,step):
    trj = []
    obs = []
    R =0
    for i,r in enumerate(rewards):
      if r!=0: #可以有其它的判别方法
        trj.append(nodes[i])
        obs.append(obses[i])
        R = R + rewards[i]
    if R>min(self.returns):
      self.trjs.append(trj)
      self.returns.append(R)
      self.obss.append(obs)
      self.max_steps.append(step)
    if len(self.returns)>self.memory_size:
      t = np.argmin(self.returns)
      del self.returns[t]
      del self.trjs[t]
      del self.obss[t]
      del self.max_steps[t]
      

  def get_keypoint(self):
    # 从buffer中找到多条好的轨迹的交叉点（由于轨迹是经过r过滤的，所以很少有重复的）
    # 由于要看中间过程，所以必须把交叉点，和交叉点对应的序号得到
    L = []
    for i,trj in enumerate(self.trjs):
      #print(trj)
      L = L + trj
    b = dict(Counter(L))
    keypoints = [key for key,value in b.items() if value>1]
    # 下面是要把这些点显示出来
    keyobss = []
    for i,kp in enumerate(keypoints):
      for j,trj in enumerate(self.trjs):
        if kp in trj:
          
          ind = trj.index(kp)
          #print("i",i,"j",j,"kp",kp,"trj",trj,"ind",ind,"obsj",self.obss)
          keyobss.append(self.obss[j][ind])

    return keypoints,keyobss
  
  
# Adapted from github.com/devsisters/DQN-tensorflow/
class ReplayMemory:
  def __init__(self, memory_size, obs_size):
    self.memory_size = memory_size
    self.obs_size = obs_size

    if self.obs_size[0] == None:
        self.observations = [None]*self.memory_size
    else:
        self.observations = np.empty([self.memory_size]+self.obs_size, dtype = np.float16)
    self.actions = np.empty(self.memory_size, dtype=np.int16)
    self.returns = np.empty(self.memory_size, dtype = np.float16)
    self.terminal = np.empty(self.memory_size, dtype = np.bool_)

    self.count = 0
    self.current = 0

  def add(self, obs, action, returns, terminal):
    self.observations[self.current] = obs
    self.actions[self.current] = action
    self.returns[self.current] = returns
    self.terminal[self.current] = terminal

    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def _get_state(self, index, seq_len):
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    if seq_len == 0:
      state = self.observations[index]
    else:
      if self.obs_size[0] == None:
        state = []
        for i in range(seq_len):
          state.append(self.observations[index-i])
      else:
        state = np.zeros([seq_len]+self.obs_size)
        for i in range(seq_len):
          state[i] = self.observations[index-i]
    return state

  def _uninterrupted(self, start, final):
    if self.current in range(start+1, final):
        return False
    for i in range(start, final-1):
        if self.terminal[i] == True: return False
    return True

  def sample(self, batch_size, seq_len=0):
    # sample random indexes
    indexes = [] ; states = []
    watchdog = 0
    while len(indexes) < batch_size:
      while True:
        # find random index
        index = np.random.randint(1, self.count - 1)
        if seq_len is not 0:
          start = index-seq_len
          if not self._uninterrupted(start, index):
            continue
        break

      indexes.append(index)
      states.append(self._get_state(index, seq_len))

    return states, self.actions[indexes], self.returns[indexes]

# Preprocessors:
def default_preprocessor(state):
    return state

def greyscale_preprocessor(state):
    #state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)/255.
    state = np.dot(state[...,:3], [0.299, 0.587, 0.114])
    return state

def deepmind_preprocessor(state):
    state = greyscale_preprocessor(state)
    #state = np.array(cv2.resize(state, (84, 84)))
    #resized_screen = scipy.misc.imresize(state, (110,84))
    resized_screen = resize(state, (110,84))
    state = resized_screen[18:102, :]
    return state

