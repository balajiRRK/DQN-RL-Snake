# Deep Reinforcement Learning Snake Game AI

This project implements a Deep Q-Network with a Convolutional Neural Network to learn and play the classic Snake game. Built with PyTorch and Pygame, the agent learns through interaction with the Snake game environment through repetitive play where it fine-tunes its policy to achieve better results. The training process includes model checkpointing, gameplay recording, epsilon-greedy exploration, logging key metrics and performance visualization with rolling averages of score and loss.

There are two pre-trained model files included: `best_model.pth`, which contains only the trained model weights and `training_checkpoint.pth`, which includes the model weights along with additional training data to allow you to resume training. Both were trained using Google Colab's Cloud NVIDIA L4 Tensor Core GPU for ~6 hours over 20,000 episodes, achieving an average score of ~35 and a peak score of 60.

Below is a visualization of key metrics logged during training that the program creates once the training is completed:
<img width="1000" height="800" alt="training_metrics" src="https://github.com/user-attachments/assets/b59b8fe0-fa4e-4aa2-b8d4-b9f5fdc7ea01" />

## Installation

1. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

To train a new model:
```
python snake_agent.py train
```

To continue training a saved model:
```
python snake_agent.py train <model_weights_filename>.pth
```

To test a saved model:
```
python snake_agent.py test <model_weights_filename>.pth
```

Additionally, you can also modify the hyperparameters or constants in `snake_agent.py` file such as RENDER_EVERY to either speed up training by reducing the frequency of rendering being done or you can slow down training but improve visualization potentially for debugging purposes by increasing the frequency of rendering. 

## Demo

https://github.com/user-attachments/assets/50e42d60-d548-4a0e-9111-fd0cfd1d885d
