# coding: utf-8

import os,glob,sys,random
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
from multiprocessing import Process, Queue
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

color_dir = './class_col/'
new_color_dir = './new_class_col/'

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


def func(num,readQ):
    print(f'--Process({num}) start')
    cnt = 0
    try:
        while True:
            img, output_path = readQ.get(timeout=20) #キューからトゥートを取り出すよー！
            img = Image.fromarray(img)
            if min(img.size) > 512:
                line = new_convert(img, "RGB")
                line.save(output_path, optimize=True)
                cnt += 1
                if cnt%100 == 0:
                    print(f'--executing image_to_line({num}) {cnt}images--')

    except:
        print(f'--Finish process {cnt}--')
        return


def reader(readQ):
    print('--Start reading--')
    cnt = 0
    for p in os.listdir(color_dir):
        for f in os.listdir(color_dir + p + '/'):
            input_path = color_dir + p + '/' + f
            output_path = new_color_dir + p + '/' + f.split(".")[0] + ".png"
            if not os.path.exists(output_path):
                img = Image.open(input_path)
                img = np.asarray(img, dtype=np.uint8)
                readQ.put((img,output_path))
                cnt += 1
                if cnt % 100 == 0:
                    print(f'reading {cnt}images...')
    print(f'--Finish reading {cnt}--')


if __name__ == "__main__":
    if not os.path.exists(color_dir):
        exit()

    if not os.path.exists(new_color_dir):
        os.mkdir(new_color_dir)

    for p in os.listdir(color_dir):
        if not os.path.exists(new_color_dir + p + '/'):
            os.mkdir(new_color_dir + p)

    readQ = Queue()

    p_r = Process(target=reader, args=(readQ,))
    p_r.start()

    p_n = []
    for num in range(0,10):
        tmp  = Process(target=func, args=(num,readQ))
        tmp.start()
        p_n.append(tmp)

