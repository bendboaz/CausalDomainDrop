import configparser
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

configs = configparser.ConfigParser()
configs.read(os.path.join(PROJECT_ROOT, "user_specific.ini"))

defaults = 'DEFAULT'
user_spec = 'USER_SPECIFIC'

DATA_ROOT = configs.get(user_spec, 'DATA_DIR')
EXPERIMENTS_ROOT = os.path.join(DATA_ROOT, os.pardir, 'experiments')

GLOVE_RAW_DIR = configs.get(user_spec, 'GLOVE_RAW_DIR')
GLOVE_PROC_DIR = configs.get(user_spec, 'GLOVE_PROC_DIR')

if configs.getboolean(user_spec, 'GLOVE_IS_RELATIVE'):
    GLOVE_PROC_DIR = os.path.join(DATA_ROOT, GLOVE_PROC_DIR)
    GLOVE_RAW_DIR = os.path.join(DATA_ROOT, GLOVE_RAW_DIR)

# if not os.path.exists(PROC_REVIEWS_PATH):
#     os.mkdir(PROC_REVIEWS_PATH)
