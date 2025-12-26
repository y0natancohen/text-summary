import json
import numpy as np
from diskcache import Cache


DATA_SET_PATH = "/home/jonathan/PycharmProjects/tavily/summaries_1k.json"
cache = Cache("~/PycharmProjects/tavily/.diskcache")

@cache.memoize()
def load_data_set():
    with open(DATA_SET_PATH, "r") as fp:
        data_set = json.load(fp)['data']
    return data_set
DATA_SET = load_data_set()

np.random.seed(1)
np.random.shuffle(DATA_SET)
TRAIN_RATIO = 0.7
TRAIN_DATA_SET = DATA_SET[:int(TRAIN_RATIO * len(DATA_SET))]
TEST_DATA_SET = DATA_SET[int(TRAIN_RATIO * len(DATA_SET)):]