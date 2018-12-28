# coding: utf-8

import os,glob,sys,random
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
from pprint import pprint as pp
ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_DIR = './temp2/'
SAVE_DIR = './gifanime/'
TARGET_FILE = '11.jpg'
Colors = {}
Colors['red']    = []
Colors['blue']   = []
Colors['green']  = []
Colors['purple'] = []
Colors['brown']  = []
Colors['pink']   = []
Colors['blonde'] = []
Colors['white']  = []
Colors['black']  = []

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        exit()

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    image_dir_list = [INPUT_DIR + p + "/" for p in os.listdir(INPUT_DIR)]
    image_dir_list.sort()

    for p in image_dir_list:
        for f in os.listdir(p):
            for c in Colors.keys():
                if c in f and TARGET_FILE.split(".")[0] + "_" + c + "." + TARGET_FILE.split(".")[-1] in f:
                    Colors[c].append(p + f)

    ani_images = []
    for i in range(len(Colors['red'])):
        base_image = Image.new("RGB", (128*3, 128*3), (0,0,0))
        for j,(c,f) in enumerate(Colors.items()):
            img = Image.open(Colors[c][i])
            # タイル状に９枚貼り付ける感じ
            x = j%3
            y = j//3
            base_image.paste(img, (x*128, y*128))

        ani_images.append(base_image)

    ani_images[0].save(f'{SAVE_DIR}anime.gif',
               save_all=True, append_images=ani_images[1:], optimize=False, duration=250, loop=0)
