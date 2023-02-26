from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT
from feature_prep import crop_clean_state

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(50000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    state = crop_clean_state(state)
    #break
    env.render()

env.close()
