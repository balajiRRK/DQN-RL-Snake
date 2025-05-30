# An RL trained Snake Agent

This agent was trained using a GTX 1070 for ~16 minutes for 2000 episodes to achieve an average score of ~20 near the 2000 episode mark and a peak of 43.

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
py train_agent.py {INSERT_NAME.pth}
```

Additionally, you can also modify the hyperparameters or constants in `train_agent.py` file such as RENDER_EVERY to either speed up training by reducing the frequency of rendering being done or you can slow down training but improve visualization potentially for debugging purposes by increasing the frequency of rendering. 

## Demo

