from pathlib import Path

RAW_DATA_DIR = Path(r"c:\Users\felip\OneDrive\Documentos\GitHub\ActiveSemisupervisedLearningCNN-s\data\train\PetImages") 
BASE_SPLIT_DIR = Path(r"c:\Users\felip\OneDrive\Documentos\GitHub\ActiveSemisupervisedLearningCNN-s\data") 

CAT_DIR_NAME = "Cat"
DOG_DIR_NAME = "Dog"

CAT_DIR = RAW_DATA_DIR / CAT_DIR_NAME
DOG_DIR = RAW_DATA_DIR / DOG_DIR_NAME


TEST_SIZE = 0.2
VAL_SIZE_RATIO = 0.125
RANDOM_SEED = 42

LABEL_CAT = 0
LABEL_DOG = 1

CLASS_NAMES = ['Cat', 'Dog']

H = 180
W = 180

NUM_CLASSES = 2