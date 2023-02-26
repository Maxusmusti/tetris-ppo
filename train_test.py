from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT
from feature_prep import crop_clean_state, extra_feats
from ppo import Controller
from ppo import PPOAgent
import tensorflow as tf
import numpy as np
import time
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Remove to regain GPU ability
tf.compat.v1.disable_eager_execution()

# Number of possible actions that PPO Agent can take
act_space_size = 6

def train(agent, epochs, batch_steps, episode_steps):
    """
    Trains two PPO agents with specified number of epochs, batch_steps, and episode_steps
    """
    final_out = ""
    lll = []

    # use agent1's controller as the main environment controller
    controller = agent.controller
    env = agent.env 
    done = True

    for epoch in range(epochs):
        print(epoch)
        st = time.perf_counter()
        ll = []

        #if epoch % 50 == 1:
            #save_model(agent, str(epoch))

        while len(controller.X1) < batch_steps:
            print(len(controller.X1))
            # reset the environment
            if done:
                obs = env.reset()

            # Get raw observation and create new observation vector
            raw_obs = obs
            cleaned_obs = crop_clean_state(raw_obs)
            info_vec = np.zeros(shape=(1,5))

            rews = []
            steps = 0
            while True:
                steps += 1

                # Prediction, action, save prediction
                agent_pred, agent_act = [x[0] for x in agent.controller.pf([cleaned_obs[None], info_vec])]
                agent.controller.P.append(agent_pred)

                # Add a decaying randomness to the chosen action
                probability = 1 - (10*epoch)/epochs
                probability = 0 if probability < 0 else probability
                if np.random.random_sample() < probability:
                    agent_act = np.random.choice(act_space_size)

                # Save this state action pair
                agent.save_pair(cleaned_obs, info_vec, agent_act)

                # Take the action and save the reward
                obs, agent_rew, done, info = env.step(agent_act)
                #env.render()
                # Take bonus steps to simplify:
                if not done:
                    for i in range(10):
                        obs, fake_rew, done, info = env.step(0)
                        #env.render()
                        if fake_rew > agent_rew:
                            agent_rew = fake_rew
                        if done:
                            break
                    

                raw_obs = obs
                cleaned_obs = crop_clean_state(raw_obs)
                info_vec = extra_feats(info)

                rews.append(agent_rew)

                if steps == episode_steps or done:
                    ll.append(np.sum(rews))

                    for i in range(len(rews)-2, -1, -1):
                        rews[i] += rews[i+1] * agent.controller.gamma
                    agent.controller.V.extend(rews)

                    break
        
        loss, vloss = agent.controller.fit()
        #loss, vloss = (None, None)
        #controller.X1 = []

        if loss != None and vloss != None:
            lll.append((epoch, np.mean(ll), loss, vloss, len(agent.controller.X1), len(ll), time.perf_counter() - st))
            print("%3d  ep_rew:%9.2f  loss:%7.2f   vloss:%9.2f  counts: %5d/%3d tm: %.2f s" % lll[-1])
            print("Episode No: %3d  Episode Reward: %9.2f" % (lll[-1][0], lll[-1][1]))
            sign = "+" if lll[-1][1] >= 0 else ""
            final_out += sign + str(lll[-1][1])

def save_model(agent, name):
    print()
    print('saving popt')
    tf.keras.models.save_model(agent.controller.popt, f'./saved_models/popt_{name}')
    print('saving p')
    tf.keras.models.save_model(agent.controller.p, f'./saved_models/p_{name}')
    print('saving v')
    tf.keras.models.save_model(agent.controller.v, f'./saved_models/v_{name}')

def main():
    gamma = 0.99
    epochs = 1000
    batch_steps = 1000
    episode_steps = 2000

	# Create env
    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Declare observation shape, action space and model controller
    observation_shape = (160, 80, 1)
    action_space = env.action_space
    controller = Controller(gamma, observation_shape, action_space, 'CONTROLLER')

    # Declare, train and run agent
    agent = PPOAgent(env, controller=controller)
    train(agent, epochs=epochs, batch_steps=batch_steps, episode_steps=episode_steps)
    save_model(agent, "trial")
    env.close()

if __name__ == "__main__":
    main()
