DATAPATH = "./dataset"
CSV_NAME = "train-metadata.csv"
IMG_DIR  = "train-image"

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.1, 0.2
USE_GROUP_SPLIT = True   # if True, split by patient_id to avoid leakage

BATCH_TRIPLET = 64
BATCH_CLASSIF = 64
NUM_WORKERS   = 4

# normalization stats
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

# training hyperparameters
TRIPLET_MARGIN = 1.0
EPOCHS_SIAMESE = 180
EPOCHS_CLS     = 140
LR_SIAMESE     = 1e-4
LR_CLS         = 5e-4

# save paths
MODELPATH = "./models"
IMAGEPATH = "./images"