import numpy as np
import os
import shutil
import PIL.ImageOps
from openslide import *
from scipy.ndimage.morphology import binary_dilation
from multiprocessing import Pool
from config import Config


def tissue_ratio(patch):
    patch = PIL.ImageOps.grayscale(patch)
    patch = PIL.ImageOps.invert(patch)
    thresholded = np.array(patch) > 100
    thresholded = binary_dilation(thresholded, iterations=15)
    ratio = np.mean(thresholded)

    return ratio, thresholded


def get_patches(slide_id, config):
    if os.path.isdir("%s/patches_%d/%s" % (config.data_path, config.patch_size, slide_id[:-4])):
        print("sample already processed")
        return

    if os.path.isdir("%s/temp/%s" % (config.data_path, slide_id[:-4])):
        shutil.rmtree("%s/temp/%s" % (config.data_path, slide_id[:-4]))
    os.makedirs("%s/temp/%s" % (config.data_path, slide_id[:-4]))
    img = OpenSlide("%s/slides/%s" % (config.data_path, slide_id))
    width, height = img.dimensions
    idx = 0
    for i in range(int(height / config.patch_size)):
        print("iteration %d out of %d" % (i + 1, int(height / config.patch_size)))
        for j in range(int(width / config.patch_size)):
            patch = img.read_region(location=(j * config.patch_size, i * config.patch_size), level=0,
                                    size=(config.patch_size, config.patch_size)).convert('RGB')
            ratio, mask = tissue_ratio(patch)
            if ratio >= config.threshold:
                patch.save("%s/temp/%s/%s.jpg" % (config.data_path, slide_id[:-4], idx))
                idx += 1
    shutil.move("%s/temp/%s" % (config.data_path, slide_id[:-4]),
                "%s/patches_%d/%s" % (config.data_path, config.patch_size, slide_id[:-4]))
    return


def get_all_patches(config, processes=10):
    patient_ids = os.listdir("%s/slides" % config.data_path)
    p = Pool(processes)
    p.starmap(get_patches, [(patient_id, config) for patient_id in patient_ids])


config = Config()
get_all_patches(config)
