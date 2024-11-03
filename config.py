import torch 

# config.py
class Config:
    BATCH_SIZE = 32
    NUM_POINTS = 1024
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    EPOCHS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
