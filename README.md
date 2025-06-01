# An RL trained Snake Agent

This project implements a Deep Q-Learning (DQN) agent to learn and play the classic Snake game. Built with PyTorch and Pygame, the agent learns through interaction with the Snake game environment through repetitive play where it fine-tunes its policy to achieve better results. The training process includes model checkpointing, gameplay recording, epsilon-greedy exploration, and performance visualization with rolling averages of score and loss.

There is an included version of a trained agent with model weights given as a .pth file which was trained using a GTX 1070 for ~1 hour for 6000 episodes to achieve an average score of ~25 near the 3000 episode mark and a peak of 73.

## Instructions to run program

1. Install dependencies:
```
pip install -r requirements.txt
```

2. You can either 
    - train a new model 
    - continue training a saved model

To train a new model:
```
py train_agent.py
```

To continue training a saved model:
```
py train_agent.py <model_weights_filename>.pth
```

Additionally, you can also modify the hyperparameters or constants in `train_agent.py` file such as RENDER_EVERY to either speed up training by reducing the frequency of rendering being done or you can slow down training but improve visualization potentially for debugging purposes by increasing the frequency of rendering. 

## Demo

https://github.com/user-attachments/assets/50e42d60-d548-4a0e-9111-fd0cfd1d885d



