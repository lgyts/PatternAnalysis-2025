# params.py
# Configuration parameters for Siamese network training and evaluation.
# Author: s4778251

# Dataset and Path Settings 
DATAPATH = "./dataset"             
CSV_NAME = "train-metadata.csv"    
IMG_DIR = "train-image"            

MODELPATH = "./models"             
IMAGEPATH = "./images"             

# Data Split and Loader 
SEED = 42                          # one true number!
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.1, 0.2   # Dataset split ratios
USE_GROUP_SPLIT = True             # Whether to use patient-based group split

BATCH_TRIPLET = 64                 # Batch size for Siamese training (triplet loss)
BATCH_CLASSIF = 64                 # Batch size for classifier training
NUM_WORKERS = 4                    # Number of worker threads for data loading

# Image Preprocessing
MEAN = [0.5, 0.5, 0.5]             
STD  = [0.5, 0.5, 0.5]             
IMAGE_SIZE = 256                   # Image resize dimension
ROT_DEG = 15                       # Max rotation degree for data augmentation
FLIP_PROB = 0.5                    # Probability of horizontal/vertical flip
COLOR_JITTER = dict(               #  color jitter parameters
    brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02
)

# Model Settings 
OUT_DIM = 512                      # Output embedding dimension of Siamese encoder
HIDDEN_DIMS = (256, 64)            # Hidden layer dimensions for classifier MLP
NEGATIVE_SLOPE = 0.01              # LeakyReLU slope for classifier
DROPOUT_P = 0.4                    # Dropout probability for classifier layers

# Training Hyperparameters
TRIPLET_MARGIN = 1.0               # Margin for triplet loss
EPOCHS_SIAMESE = 100               # Max epochs for Siamese encoder training
EPOCHS_CLS = 80                    # Max epochs for classifier training
LR_SIAMESE = 0.0001                # Learning rate for Siamese encoder
LR_CLS = 0.0005                    # Learning rate for classifier

# Early Stop / Scheduler
PATIENCE = 5                       # Early stopping patience in epochs
MIN_DELTA = 0.001                  # Minimum improvement threshold for validation loss
SCHED_FACTOR = 0.5                 
SCHED_PATIENCE = 3                

# Output Filenames
SAVE_SAMPLE_NAME = "input_sample.png"   # Example image filename
SIAMESE_LOSS_NAME = "siamese_loss.png"  # Siamese loss plot filename
CLS_LOSS_NAME = "classifier_loss.png"   # Classifier loss plot filename
CM_NAME = "confusion_matrix.png"        # Confusion matrix filename
