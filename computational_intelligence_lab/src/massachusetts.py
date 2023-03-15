import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt 


DATA_PATH = "data/processed/massachusetts/groundtruth"

def func():

    size = len(os.listdir(DATA_PATH))
    percentages = np.zeros(size)


    for index, name in tqdm(enumerate(os.listdir(DATA_PATH))):
        path = os.path.join(DATA_PATH, name)
        im = Image.open(path)
        im = np.asarray(im)
        percentages[index] = np.count_nonzero(im) / im.size

    print(np.percentile(percentages, 10))
    print(np.percentile(percentages, 25))
    print(np.percentile(percentages, 50))
    print(np.percentile(percentages, 75))
    print(np.percentile(percentages, 90))

    plt.hist(percentages, bins=100) 
    plt.title("histogram") 
    plt.show()

if __name__ == '__main__':
    func()