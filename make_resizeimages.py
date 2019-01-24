# coding: utf-8

import os,glob,sys,random
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
from multiprocessing import Process, Queue
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

color_dir = './class_col/'
resize_dir = './class_col_rs/'

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
    tmp = img.copy()
    if tmp.mode == 'L':
        tmp = expand2square(tmp,(255,))
    else:
        tmp = expand2square(tmp,(255,255,255))
    return tmp.resize(resize,Image.BICUBIC)


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


def image_to_line_prc(num,readQ):
    print(f'--Process({num}) start')
    cnt = 0
    try:
        while True:
            img, output_path = readQ.get(timeout=10) #キューからトゥートを取り出すよー！
            img = Image.fromarray(img)
            img = new_convert(img, "RGB")
            img = image_resize(img, (512,512))
            img.save(output_path, optimize=True)
            cnt += 1
            if cnt%100 == 0:
                print(f'--executing image_to_line({num}) {cnt}images--')
    except Exception as e:
        print(e)
        print(f'--Finish image_to_line({num}) {cnt}images--')
        return

def reader(readQ):
    print('--Start reading--')
    cnt = 0
    for p in os.listdir(color_dir):
        for f in os.listdir(color_dir + p + '/'):
            input_path = color_dir + p + '/' + f
            output_path = resize_dir + p + '/' + f
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

    if not os.path.exists(resize_dir):
        os.mkdir(resize_dir)

    for p in os.listdir(color_dir):
        if not os.path.exists(resize_dir + p + '/'):
            os.mkdir(resize_dir + p)

    readQ = Queue()

    p_r = Process(target=reader, args=(readQ,))
    p_r.start()

    p_n = []
    for num in range(0,10):
        tmp  = Process(target=image_to_line_prc, args=(num,readQ))
        tmp.start()
        p_n.append(tmp)
   
