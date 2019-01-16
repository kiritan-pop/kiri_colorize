# coding: utf-8
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend

import os,glob,sys,json,random
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

STANDARD_SIZE_S1 = (128, 128)
STANDARD_SIZE_S2 = (512, 512)

Colors = {}
Colors['red']    = 0
Colors['blue']   = 1
Colors['green']  = 2
Colors['purple'] = 3
Colors['brown']  = 4
Colors['pink']   = 5
Colors['blonde'] = 6
Colors['white']  = 7
Colors['black']  = 8

Colors_rev = {v:k for k,v in Colors.items()}

total_epochs = 999999
STANDARD_SIZE = (299,299)
input_dir = './line_images/'
results_dir = './gen1_images/'
g_model_s1_path = 'g_model_s1.h5'
test_dir = './test/'
test_colored_dir = './test_colored/'
test_short_dir = './test_short/'
test_short_colored_dir = './test_short_colored/'

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def image_resize(img, resize):
    # アスペクト比維持
    tmp = img
    # tmp.thumbnail(resize,Image.BICUBIC)
    if tmp.mode == 'L':
        tmp = expand2square(tmp,(255,))
    else:
        tmp = expand2square(tmp,(255,255,255))

    return tmp.resize(resize,Image.BICUBIC)

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def new_convert(img, mode):
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
    elif img.mode == "LA":
        bg = Image.new("L", img.size, (255,))
        bg.paste(img, mask=img.split()[1])
    else:
        bg = img
    return bg.convert(mode)


if __name__ == "__main__":
    if not os.path.exists(input_dir):
        exit()

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    for p in os.listdir(input_dir):
        if not os.path.exists(results_dir + p + '/'):
            os.mkdir(results_dir + p)

    #GPU設定
    config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False, visible_device_list="2"),
                allow_soft_placement=True, 
                log_device_placement=False
                )
    session = tf.Session(config=config)
    backend.set_session(session)

    gens1_model = load_model(g_model_s1_path)

    for col in os.listdir(input_dir):
        for f in os.listdir(input_dir + col):
            line_image_path = input_dir + col + "/" + f
            print(line_image_path)
            filename = results_dir + col + "/" + f
            # if os.path.exists(filename):
            #     continue

            img = Image.open(line_image_path)
            img = new_convert(img, "L")

            line_s1 = image_resize(img, STANDARD_SIZE_S1)
            line_s1 = (np.asarray(line_s1)-127.5)/127.5
            line_s1 = line_s1.reshape((1,128,128))

            color_name = []
            color_val = np.zeros((1,), dtype=np.uint8)
            color_val[0] = Colors[col]

            ret = gens1_model.predict_on_batch([line_s1, color_val])

            tmp = (ret[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
            tmp = Image.fromarray(tmp)
            tmp.save(filename, optimize=True)


    for dir,outdir in zip([test_dir,test_short_dir],[test_colored_dir,test_short_colored_dir]):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for f in os.listdir(dir):
            img = Image.open(dir+f).convert('L')
            tmp = expand2square(img,(255,))
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
                tmp.save(filename, optimize=True)
