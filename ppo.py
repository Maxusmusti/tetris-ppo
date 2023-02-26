"""Full game environment implementing PPO for a 1v1 game"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Concatenate, AveragePooling2D

tf.compat.v1.disable_eager_execution()

#import torch
#import torch.nn as nn

"""
class VCritic(nn.Module):
    def __init__(self):
        super(VCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # adjust the input size to match the output of the last pooling layer
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)  # flatten the output of the last convolutional layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class PActor(nn.Module):
    def __init__(self, action_space):
        super(PActor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # adjust the input size to match the output of the last pooling layer
        self.fc2 = nn.Linear(128, action_space)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)  # flatten the output of the last convolutional layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
"""

class Controller(object):
    def __init__(self,
                 gamma=0.99,
                 observation_shape=None,
                 action_space=None,
                 name='agent'):

        self.gamma = gamma

        self.observation_shape = observation_shape
        self.action_space = action_space

        self.init_policy_function()
        self.init_value_function()
        self.pretrained = self.load_pretrained()

        self.X1 = []
        self.X2 = []
        self.Y = []
        self.V = []
        self.P = []

        self.n_agents = 0
        self.d_agents = 0
        self.cur_updating = True

        self.name = name

    def load_pretrained(self):
        return None

    def init_value_function(self):
        # value function
        x = in1 = Input(self.observation_shape)
        in2 = Input((5,))
        #x = Conv2D(filters=16, kernel_size=(4, 4), strides=(4,4), activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        x = st = MaxPooling2D(pool_size=(8,8), strides=(8,8))(x)
        x = Conv2D(filters=1, kernel_size=(3, 3), activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        #x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        st = Flatten()(st)
        x = Flatten()(x)
        x = Concatenate()([x, st, in2])
        x = Dense(units=32, activation='relu')(x)
        x = Dense(units=32, activation='relu')(x)
        x = Dense(units=1, activation='sigmoid')(x)
        v = Model(inputs=[in1, in2], outputs=x)

        v.compile(Adam(1e-3), 'binary_crossentropy')
        v.summary()

        # May need adjustment
        vf = K.function(inputs=[in1, in2], outputs=v.layers[-1].output)

        self.vf = vf
        self.v = v

    def init_policy_function(self):
        action_space = self.action_space

        # policy function
        x = in1 = Input(self.observation_shape)
        in2 = Input((5,))
        #x = Conv2D(filters=16, kernel_size=(4, 4), strides=(4,4), activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        x = st = MaxPooling2D(pool_size=(8,8), strides=(8,8))(x)
        x = Conv2D(filters=1, kernel_size=(3, 3), activation='relu')(x)
        #x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        #x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        st = Flatten()(st)
        x = Flatten()(x)
        x = Concatenate()([x, st, in2])
        x = Dense(units=32, activation='relu')(x)
        x = Dense(units=32, activation='relu')(x)
        x = Dense(action_space.n)(x)
        action_dist = Lambda(lambda x: tf.nn.log_softmax(x, axis=-1))(x)
        p = Model(inputs=[in1, in2], outputs=action_dist)
        
        in_advantage = Input((1,))
        in_old_prediction = Input((action_space.n,))

        def loss(y_true, y_pred):
            advantage = tf.reshape(in_advantage, (-1,))
        
            # y_pred is the log probs of the actions
            # y_true is the action mask
            prob = tf.reduce_sum(y_true * y_pred, axis=-1)
            old_prob = tf.reduce_sum(y_true * in_old_prediction, axis=-1)
            ratio = tf.exp(prob - old_prob)
            
            # this is the VPG objective
            #ll = -(prob * advantage)
            
            # this is PPO objective
            ll = -K.minimum(ratio*advantage, K.clip(ratio, 0.8, 1.2)*advantage)
            return ll

        # To lookout for
        popt = Model([in1, in2, in_advantage, in_old_prediction], action_dist)
        popt.compile(Adam(5e-4), loss)
        popt.summary()

        # May need adjustment
        pf = K.function(inputs=[in1, in2],
                        outputs=[p.layers[-1].output,
                        tf.random.categorical(p.layers[-1].output, 1)[0]])
                        
        self.pf = pf
        self.popt = popt
        self.p = p

    def fit(self, batch_size=5, epochs=10, shuffle=True, verbose=0):
        X1 = self.X1
        X2 = self.X2
        Y = self.Y
        V = self.V
        P = self.P

        self.d_agents += 1 

        if self.d_agents < self.n_agents:
            return None, None

        print("[FIT] TRAINING ON DATA FOR", self.name)
        X1, X2, Y, V, P = [np.array(x) for x in [X1, X2, Y, V, P]]
        X2 = np.squeeze(X2, axis=1)

        # Subtract value baseline to get advantage
        A = V - self.vf([X1, X2])[:, 0]

        loss = self.popt.fit([X1, X2, A, P], Y, batch_size=5, epochs=10, shuffle=True, verbose=0)
        loss = loss.history["loss"][-1]
        vloss = self.v.fit([X1, X2], V, batch_size=5, epochs=10, shuffle=True, verbose=0)
        vloss = vloss.history["loss"][-1]

        self.X1 = []
        self.X2 = []
        self.Y = []
        self.V = []
        self.P = []
        
        self.d_agents = 0

        return loss, vloss

    #def get_pred_act(self, obs):
    #    pred, act = [x[0] for x in self.pf(obs[None])]
    #    return pred, act

    def register_agent(self):
        self.n_agents += 1
        return self.n_agents

class PPOAgent(object):
    """Basic PPO implementation for LoLGym environment."""
    def __init__(self, env, controller=None):
        if not controller:
            raise ValueError("PPOAgent needs to be provided an external controller")
        
        self.controller = controller
        self.agent_id = controller.register_agent()

        print("PPOAgent:", self.agent_id, "Controller:", self.controller)
        self.env = env

    def save_pair(self, obs, vec, act):
        action_space = self.controller.action_space
        self.controller.X1.append(np.copy(obs))
        self.controller.X2.append(np.copy(vec))
        act_mask = np.zeros((action_space.n))
        act_mask[act] = 1.0
        self.controller.Y.append(act_mask)

val = inp = Input((160,80,1))
val = MaxPooling2D(pool_size=(8,8), strides=(8,8))(val)
state_model = Model(inputs=inp, outputs=val)
state_model.compile(loss='mse')
sf = K.function(inputs=inp,outputs=val)

def compute_reward(obs, rew):
    try:
        state = np.squeeze(np.squeeze(sf(np.expand_dims(obs, axis=0)), axis=0), axis=2)
    except:
        print("bad state")
        return rew
    return rew
