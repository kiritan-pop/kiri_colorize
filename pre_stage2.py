# coding: utf-8
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend

import os,glob,sys,json,random
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter
import argparse

from kiri_datagenerator import Colors, STANDARD_SIZE_S1, STANDARD_SIZE_S2, Colors_rev, expand2square


ImageFile.LOAD_TRUNCATED_IMAGES = True

total_epochs = 999999

test_dir = './test/'
test_colored_dir = './test_colored/'
test_short_dir = './test_short/'
test_short_colored_dir = './test_short_colored/'
g_model_s1_path = 'g_model_s1.h5'

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


if __name__ == "__main__":
    #GPU設定
    config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False, visible_device_list="2"),
                allow_soft_placement=True, 
                log_device_placement=False
                )
    session = tf.Session(config=config)
    backend.set_session(session)

    gens1_model = load_model(g_model_s1_path)

    for dir,outdir in zip([test_dir,test_short_dir],[test_colored_dir,test_short_colored_dir]):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for f in os.listdir(dir):
            img = Image.open(dir+f).convert('RGB')
            tmp = expand2square(img,(255,255,255))
            line_s1 = tmp.resize(STANDARD_SIZE_S1,Image.BICUBIC)
            line_s1 = (np.asarray(line_s1)-127.5)/127.5

            for color,val in Colors.items():
                save_path = outdir + color + '/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                ret = gens1_model.predict_on_batch([np.asarray([line_s1]), np.asarray([val])])

                filename = save_path + f
                tmp = (ret[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                tmp = Image.fromarray(tmp)
                tmp = crop_center(tmp, img.width, img.height)
                tmp.save(filename, optimize=True)









