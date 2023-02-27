"""
Full game environment implementing PPO for Tetris
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Lambda,
    Concatenate,
)

tf.compat.v1.disable_eager_execution()


class Controller(object):
    def __init__(
        self, gamma=0.99, observation_shape=None, action_space=None, name="agent"
    ):

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
        # x = Conv2D(filters=16, kernel_size=(4, 4), strides=(4,4), activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        x = st = MaxPooling2D(pool_size=(8, 8), strides=(8, 8))(x)
        x = Conv2D(filters=1, kernel_size=(3, 3), activation="relu")(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        st = Flatten()(st)
        x = Flatten()(x)
        x = Concatenate()([x, st, in2])
        x = Dense(units=32, activation="relu")(x)
        x = Dense(units=32, activation="relu")(x)
        x = Dense(units=1, activation="sigmoid")(x)
        v = Model(inputs=[in1, in2], outputs=x)

        v.compile(Adam(1e-3), "binary_crossentropy")
        v.summary()

        vf = K.function(inputs=[in1, in2], outputs=v.layers[-1].output)

        self.vf = vf
        self.v = v

    def init_policy_function(self):
        action_space = self.action_space

        # policy function
        x = in1 = Input(self.observation_shape)
        in2 = Input((5,))
        # x = Conv2D(filters=16, kernel_size=(4, 4), strides=(4,4), activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        x = st = MaxPooling2D(pool_size=(8, 8), strides=(8, 8))(x)
        x = Conv2D(filters=1, kernel_size=(3, 3), activation="relu")(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        st = Flatten()(st)
        x = Flatten()(x)
        x = Concatenate()([x, st, in2])
        x = Dense(units=32, activation="relu")(x)
        x = Dense(units=32, activation="relu")(x)
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
            # ll = -(prob * advantage)

            # this is PPO objective
            ll = -K.minimum(ratio * advantage, K.clip(ratio, 0.8, 1.2) * advantage)
            return ll

        popt = Model([in1, in2, in_advantage, in_old_prediction], action_dist)
        popt.compile(Adam(5e-4), loss)
        popt.summary()

        pf = K.function(
            inputs=[in1, in2],
            outputs=[
                p.layers[-1].output,
                tf.random.categorical(p.layers[-1].output, 1)[0],
            ],
        )

        self.pf = pf
        self.popt = popt
        self.p = p

    def fit(self, batch_size=5, epochs=5, shuffle=True, verbose=0):
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

        loss = self.popt.fit(
            [X1, X2, A, P], Y, batch_size=5, epochs=5, shuffle=True, verbose=0
        )
        loss = loss.history["loss"][-1]
        vloss = self.v.fit([X1, X2], V, batch_size=5, epochs=5, shuffle=True, verbose=0)
        vloss = vloss.history["loss"][-1]

        self.X1 = []
        self.X2 = []
        self.Y = []
        self.V = []
        self.P = []

        self.d_agents = 0

        return loss, vloss

    def register_agent(self):
        self.n_agents += 1
        return self.n_agents


class TestController(object):
    def __init__(
        self,
        gamma=0.99,
        observation_shape=None,
        action_space=None,
        name="agent",
        v=None,
        p=None,
        popt=None,
    ):

        self.gamma = gamma

        self.observation_shape = observation_shape
        self.action_space = action_space

        self.p = p
        self.popt = popt
        self.init_policy_function()
        self.v = v
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
        vf = K.function(
            inputs=[self.v.layers[0].input, self.v.layers[5].input],
            outputs=self.v.layers[-1].output,
        )
        self.vf = vf

    def init_policy_function(self):
        # policy function
        pf = K.function(
            inputs=[self.p.layers[0].input, self.p.layers[5].input],
            outputs=[
                self.p.layers[-1].output,
                tf.random.categorical(self.p.layers[-1].output, 1)[0],
            ],
        )
        self.pf = pf

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


val = inp = Input((160, 80, 1))
val = MaxPooling2D(pool_size=(8, 8), strides=(8, 8))(val)
state_model = Model(inputs=inp, outputs=val)
state_model.compile(loss="mse")
sf = K.function(inputs=inp, outputs=val)


def bumps(map):
    bumps = 0
    pad_map = np.pad(map, [(1, 1), (1, 1)], mode="constant", constant_values=1)
    for i in range(1, len(map) + 1):
        for j in range(1, len(map[0]) + 1):
            space = 0
            if pad_map[i][j] == 1:
                if pad_map[i + 1][j] == 0:
                    space += 1
                if pad_map[i - 1][j] == 0:
                    space += 1
                if pad_map[i][j + 1] == 0:
                    space += 1
                if pad_map[i][j - 1] == 0:
                    space += 1
            if space >= 3:
                bumps += 1
    return bumps


def holes(map):
    holes = 0
    pad_map = np.pad(map, [(1, 1), (1, 1)], mode="constant", constant_values=1)
    for i in range(1, len(map) + 1):
        for j in range(1, len(map[0]) + 1):
            if pad_map[i][j] == 0:
                if (
                    pad_map[i + 1][j] == 1
                    and pad_map[i - 1][j] == 1
                    and pad_map[i][j + 1] == 1
                    and pad_map[i][j - 1] == 1
                ):
                    holes += 1
    return holes


def compute_reward(obs, rew, prev_holes, prev_bumps):
    try:
        state = np.squeeze(np.squeeze(sf(np.expand_dims(obs, axis=0)), axis=0), axis=2)
    except:
        print("bad state")
        return rew
    new_holes = holes(state)
    new_bumps = bumps(state)
    rew = rew + 2 + 3 * (prev_holes - new_holes) + 3 * (prev_bumps - new_bumps)
    return rew, new_holes, new_bumps
