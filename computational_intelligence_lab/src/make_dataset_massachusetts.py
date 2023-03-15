import logging
import os
import shutil
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from kaggle.api.kaggle_api_extended import KaggleApi

class MassachusettsDataset():
    def __init__(self, data_path, dataset_name="massachusetts"):
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        self.logger = logging.getLogger(__name__)

        self.raw_data_path = os.path.join(data_path, "raw")
        self.processed_data_path = os.path.join(data_path, "processed")
        self.raw_image_path = os.path.join(self.processed_data_path, "tiff")
        self.image_path = os.path.join(self.raw_image_path, "image")
        self.groundtruth_path = os.path.join(self.raw_image_path, "groundtruth")

        self.dataset_name = dataset_name
        self.final_image_path = os.path.join(self.processed_data_path, self.dataset_name, "images")
        self.final_groundtruth_path = os.path.join(self.processed_data_path, self.dataset_name, "groundtruth")

    def download(self):
        Path(self.raw_data_path).mkdir(parents=True, exist_ok=True)
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
        zip_file = os.path.join(self.raw_data_path, 'massachusetts-roads-dataset.zip')
        if os.path.isfile(zip_file):
            self.logger.info("Skip download as file is already downloaded")
        else:
            self.logger.info("Download dataset to %s", self.raw_data_path)
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files('balraj98/massachusetts-roads-dataset', path=self.raw_data_path)

        self.logger.info("Extract files to %s", self.processed_data_path)
        with zipfile.ZipFile(zip_file, 'r') as f:
            for member in f.namelist():
                member_path = os.path.join(self.processed_data_path, member)
                if not os.path.exists(member_path) or os.path.isfile(member_path):
                    f.extract(member, self.processed_data_path)

    def process_images(self, road_threshold = 0.1, white_threshold=0.3, remove_intermediate_files=False, force=False):
        self.logger.info("Convert images from .tiff to .png")
        self._convert_images()

        self.logger.info("Crop images")
        self._crop_images(road_threshold=road_threshold, white_threshold=white_threshold, force=force)

        if remove_intermediate_files:
            self.logger.info("Remove uncropped images")
            shutil.rmtree(self.raw_image_path)


    def _convert_images(self):
        Path(self.image_path).mkdir(parents=True, exist_ok=True)
        Path(self.groundtruth_path).mkdir(parents=True, exist_ok=True)

        if len(os.listdir(self.image_path)) > 0 or len(os.listdir(self.groundtruth_path)) > 0:
            self.logger.info("Skip converting images as images are already converted")
            return

        for d in os.listdir(self.raw_image_path):
            img_path = os.path.join(self.raw_image_path, d)
            self.logger.info("Convert files in path %s", img_path)
            for name in tqdm(os.listdir(img_path)):
                if os.path.splitext(os.path.join(img_path, name))[1].lower() == ".tiff" or os.path.splitext(os.path.join(img_path, name))[1].lower() == ".tif":
                    if not os.path.isfile(os.path.splitext(os.path.join(img_path, name))[0] + ".png"):
                        if(img_path.endswith("labels")):
                            path_out = os.path.join(self.groundtruth_path, name)
                        else:
                            path_out = os.path.join(self.image_path, name)
                            
                        outfile = os.path.splitext(path_out)[0] + ".png"
                        im = Image.open(os.path.join(img_path, name))
                        im.save(outfile, "png", quality=100)
                        im.close()

                    os.remove(os.path.join(img_path, name))

    def _crop_images(self, road_threshold, white_threshold, force):
        def road_percentage(groundtruth):
            gt_np = np.asarray(groundtruth)
            return np.count_nonzero(gt_np) / gt_np.size

        def whitespace_percentage(image):
            img_np = np.asarray(image)
            return np.count_nonzero(np.all(img_np==[255,255,255], axis=2)) / img_np.size


        if force:
            shutil.rmtree(self.final_image_path)
            shutil.rmtree(self.final_groundtruth_path)

        Path(self.final_image_path).mkdir(parents=True, exist_ok=True)
        Path(self.final_groundtruth_path).mkdir(parents=True, exist_ok=True)

        if (len(os.listdir(self.final_image_path)) > 0 or len(os.listdir(self.final_groundtruth_path)) > 0):
            self.logger.info("Skip cropping images as images are already cropped")
            return


        index = 0
        image_path = os.path.join(self.final_image_path, "satimage_")
        groundtruth_path = os.path.join(self.final_groundtruth_path, "satimage_")
        for name in tqdm(os.listdir(self.image_path)):
            image_in = os.path.join(self.image_path, name)
            image = Image.open(image_in)
            image = image.resize(size=(3200, 3200))
            groundtruth_in = os.path.join(self.groundtruth_path, name)
            groundtruth = Image.open(groundtruth_in)
            groundtruth = groundtruth.resize(size=(3200, 3200), resample=Image.NEAREST)

            for i in range(0, 3200, 400):
                for j in range(0, 3200, 400):
                    partial_gt = groundtruth.crop((i, j, i+400, j+400))
                    if partial_gt.getbbox():
                        partial_img = image.crop((i, j, i+400, j+400))
                        if road_percentage(partial_gt) > road_threshold and whitespace_percentage(partial_img) < white_threshold:
                            partial_img.save(image_path + str(index) + ".png", "png", quality=100)
                            partial_gt.save(groundtruth_path + str(index) + ".png", "png", quality=100)
                            index += 1


def main():
    """ Runs data processing scripts to turn raw data from (../external) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dataset = MassachusettsDataset("data")
    dataset.download()
    dataset.process_images(force = True)

if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main()

    