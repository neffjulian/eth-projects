import logging
import os
import zipfile
from pathlib import Path
from PIL import Image

from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
DATASET_NAME = 'cil-road-segmentation-2022'

def make_dataset_small():
    """Constructs a dataset with smaller images by splitting each image in four 200x200 images
    """
    base_path = os.path.join(PROCESSED_DATA_PATH, "training")
    image_path = os.path.join(base_path, "images")
    groundtruth_path = os.path.join(base_path, "groundtruth")
    image_path_small = os.path.join(base_path, "images_small")
    groundtruth_path_small = os.path.join(base_path, "groundtruth_small")
    Path(image_path_small).mkdir(parents=True, exist_ok=True)
    Path(groundtruth_path_small).mkdir(parents=True, exist_ok=True)

    h = 200
    w = 200

    for img_name in os.listdir(image_path):
        img = Image.open(os.path.join(image_path, img_name))
        gt = Image.open(os.path.join(groundtruth_path, img_name))

        original_width = img.size[0]
        original_height = img.size[1]
        for i in range(0, original_width, w):
            for j in range(0, original_height, h):
                max_i = min(i+w, original_width)
                max_j = min(j+h, original_height)
                box = (i,j,max_i,max_j)
                img_path = f"{os.path.join(image_path_small, img_name.split('.')[0])}_{i}_{j}.png"
                img_tile = img.crop(box).save(img_path)
                gt_path = f"{os.path.join(groundtruth_path_small, img_name.split('.')[0])}_{i}_{j}.png"
                gt_tile = gt.crop(box).save(gt_path)

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    Path(RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    logger.info('downloading dataset from kaggle')
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files('cil-road-segmentation-2022', path=RAW_DATA_PATH)

    logger.info('extracting dataset from zip')
    zip_file = os.path.join(RAW_DATA_PATH, DATASET_NAME + '.zip')
    with zipfile.ZipFile(zip_file, 'r') as f:
        f.extractall(PROCESSED_DATA_PATH)

    make_dataset_small()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
