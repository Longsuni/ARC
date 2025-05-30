import numpy as np
from data_utils.utils import test_model
from data_utils.parse_args import args

city = args.city
task = args.task

city = "Chi"
task = "crime"
path = "embed_result/best_emb_{}_{}.npy".format(city,task)

if __name__ == "__main__":
    embs = np.load(path)
    _, _, r2 = test_model(city, task, embs)
