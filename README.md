# An RL trained Snake Agent

This project implements a Deep Q-Learning (DQN) agent to learn and play the classic Snake game. Built with PyTorch and Pygame, the agent learns through interaction with the Snake game environment through repetitive play where it fine-tunes its policy to achieve better results. The training process includes model checkpointing, gameplay recording, epsilon-greedy exploration, logging key metrics and performance visualization with rolling averages of score and loss.

There is an included version of a trained agent with model weights given as a .pth file which was trained using Google Colab's Cloud Nvidia Tesla T4 GPU for ~5 hours for 15000 episodes to achieve an average score of ~30 at the end and a peak of 57.

Below is a visualization of key metrics logged during training that the program creates once the training is completed:
<img width="1000" height="800" alt="training_metrics" src="https://github.com/user-attachments/assets/252faeab-c8b4-43c1-ab2b-8f5ad29ba47a" />

## Instructions to run program

1. Install dependencies:
```
pip install -r requirements.txt
```

2. You can either 
    - train a new model 
    - continue training a saved model
    - test a saved model

To train a new model:
```
py snake_agent.py train
```

To continue training a saved model:
```
py snake_agent.py train <model_weights_filename>.pth
```

To test a saved model:
```
py snake_agent.py test <model_weights_filename>.pth
```

Additionally, you can also modify the hyperparameters or constants in `train_agent.py` file such as RENDER_EVERY to either speed up training by reducing the frequency of rendering being done or you can slow down training but improve visualization potentially for debugging purposes by increasing the frequency of rendering. 

## Demo

https://github.com/user-attachments/assets/50e42d60-d548-4a0e-9111-fd0cfd1d885d



