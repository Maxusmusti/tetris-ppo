# tetris-ppo
Attempting to play Classic NES Tetris via policy-based reinforcement learning (PPO)

## Analysis Report
To learn more about the project goals, methods, results, and conclusions, see `report.pdf`

## Environment
Classic NES Tetris RL environment
 - The env used in this project is [gym-tetris](https://github.com/Kautenja/gym-tetris), based on [OpenAI Gym](https://www.gymlibrary.dev/)
 - Observation states consist of a raw image of the game, as well as values like `score`, `lines_cleared`, and piece info
 - This project uses the simplified action space of: [Rotate L, Rotate R, Left, Right, Down, No-Op]
 
For more information, see `report.pdf`

## Training and Testing

Requirements
 - Running this code requires `python3`
 - There are also four required packages: `numpy`, `matplotlib`, `gym-tetris`, and `tensorflow`
 - These packages can be installed via `pip`
 - For the correct versions, use the requirements file: `pip install -r requirements.txt`

Display
 - Note that without a display device, the environment cannot be rendered
 - To run display-less (or significantly speed up training), comment out all `env.render` calls in `train_test.py`
 - Note that if you keep rendering enabled, you can watch the agent play in real-time, during both training and testing

Training and Testing
 - To toggle between training and testing, set `train_system` to true or false in the `main` function of `train_test.py`
 - To run, simply run `python train_test.py`
 - Hyperparameters and final saved/loaded models name can also be set in the `main` function
 - Training will also save model checkpoints every 10 epochs
