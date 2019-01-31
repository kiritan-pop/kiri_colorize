# coding: utf-8

import os,glob,sys,json,random,cv2,threading,pprint,queue
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
import tensorflow as tf
graph = tf.get_default_graph()

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

def image_to_line_open(path):
    tmp = path.split("/")
    r = random.randint(0,100)%5
    if r == 0:
        tmp[-3] = "line_images"
    elif r == 1:
        tmp[-3] = "line_images2"
    elif r == 2:
        tmp[-3] = "line_images4"
    elif r == 3:
        tmp[-3] = "line_images5"
    else:
        tmp[-3] = "line_images3"
    line_path = "/".join(tmp)
    img = Image.open(line_path)
    img = new_convert(img, "L")
    return img

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

def image_arrange(path, resize=(128,128)):
    img = Image.open(path)
    img = new_convert(img, 'RGB')
    #画像加工ランダム値（重いので）
    rotate_rate = random.randint(0,360) #回転
    mirror =  random.randint(0,100) % 2

    #線画（線画化してからリサイズ）
    line = image_to_line_open(path)

    # アスペクト比維持するように変更
    img = img.resize(resize, Image.BICUBIC)
    img = img.rotate(rotate_rate, expand=False, resample=Image.BICUBIC)
 
    line = line.resize(resize, Image.BICUBIC)
    line = line.rotate(rotate_rate, expand=False, resample=Image.BICUBIC)

    tmp = "gen1_images/" + "/".join(path.split("/")[2:])
    if os.path.exists(tmp):
        img_small = Image.open(tmp)
        img_small = img_small.rotate(rotate_rate, expand=False, resample=Image.BICUBIC)
    else:
        img_small = None

    if mirror ==0:
        img = ImageOps.mirror(img)
        line = ImageOps.mirror(line)
        if img_small:
            img_small = ImageOps.mirror(img_small)

    col = Colors[path.split("/")[-2]]

    return img, line, img_small, col


class D_Datagenerator():
    def __init__(self, color_path, batch_size, g_model, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.color_images = []
        for p in os.listdir(color_path):
            for f in os.listdir(color_path + p + '/'):
                self.color_images.append(color_path + p + '/' + f)
        random.shuffle(self.color_images)
        self.val = val
        self.valid_images = random.sample(self.color_images,batch_size)
        for d in self.valid_images:
            self.color_images.remove(d)
        self.g_model = g_model
        self.old_fake = {}

        # エラー回避のため、１回空振りさせる
        img = np.zeros((STANDARD_SIZE_S1[0],STANDARD_SIZE_S1[1]))
        with graph.as_default():
            g_model.predict_on_batch([np.array([img]), np.array([Colors['red']])])
        sleep(2)

    def __getitem__(self, idx):
        xy  = queue.Queue()
        # データの取得実装
        if self.val == 999:
            ytmp1 = self.valid_images
        else:
            ytmp1 = self.color_images[self.batch_size*idx:self.batch_size*(idx+1)]

        def func(path):
            img, line, _ , selvec = image_arrange(path, resize=STANDARD_SIZE_S1)
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5

            if self.val == 999:
                rand = random.randint(0,1000) 
            else:
                rand = self.val

            if rand%3 == 0:
                #本物
                x = img
                retvecs = selvec
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([1.0 - r, 0.0 + r])

            elif rand%3 == 1:
                #本物だけど、色間違い
                x = img
                # カラーラベルランダム取得（正解以外）
                tmpcol = list(Colors.values())
                tmpcol.remove(selvec) 
                selvec2 = random.choice(tmpcol)
                retvecs = selvec2
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([0.0 + r, 1.0 - r])

            else:
                #偽物
                selcol3 = random.choice(list(Colors.keys()))
                selvec3 = Colors[selcol3]

                if random.randint(0,10) == 0 and path in self.old_fake:
                    #過去の生成画像を使用する
                    x = self.old_fake[path]
                else:
                    with graph.as_default():
                        ret = self.g_model.predict_on_batch([np.array([line]), np.array([selvec3])])
                    x = ret[0]

                retvecs = selvec3
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([0.0 + r, 1.0 - r])

                r = random.randint(0, self.batch_size*5)
                if r == 0:
                    filename = f'temp/gen_{selcol3}.png'
                    tmp = (x*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    filename = f'temp/gen_{selcol3}_in.png'
                    tmp = (line*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    #偽物画像を保存しておく（次のDiscriminater用の学習へ）
                    self.old_fake[path] = x

                    if len(self.old_fake) > 1000:
                        del self.old_fake[random.choice(list(self.old_fake.keys() ))]

            xy.put([x, lines, retvecs, y])

        threads = []
        for path in ytmp1:
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        lines = []
        retvecs = []
        y = []
        while xy.qsize() > 0:
            a,b,c,d = xy.get()
            x.append(a)
            lines.append(b)
            retvecs.append(c)
            y.append(d)

        return [np.asarray(x), np.asarray(retvecs)], np.asarray(y)

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        sample_per_epoch = math.ceil(len(self.color_images)/self.batch_size)
        return sample_per_epoch

class Comb_Datagenerator():
    def __init__(self, color_path, batch_size, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.val = val
        self.color_images = {}
        self.color_images_flat = []
        self.valid_images = []
        for p in os.listdir(color_path):
            tmp = []
            for f in os.listdir(color_path + p + '/'):
                tmp.append(color_path + p + '/' + f)

            valid_tmp = random.sample(tmp, math.ceil(batch_size/len(Colors)))
            self.valid_images.extend(valid_tmp)
            for d in valid_tmp:
                tmp.remove(d)
            random.shuffle(tmp)
            self.color_images[p] = tmp
            self.color_images_flat.extend(tmp)

        random.shuffle(self.color_images_flat)

    def __getitem__(self, idx):
        # データの取得実装
        if self.val == 999:
            ytmp1 = random.sample(self.valid_images, self.batch_size)
        else:
            ytmp1 = self.color_images_flat[self.batch_size*idx:self.batch_size*(idx+1)]
        xy  = queue.Queue()
        def func(path):
            img, line, _, selvec = image_arrange(path, resize=STANDARD_SIZE_S1)
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5

            r = random.uniform(0.0, 0.2)
            y = [1.0 - r, 0.0 + r]

            xy.put([line, selvec, y, img])

        threads = []
        for path in ytmp1:
            #本物
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        selvecs = []
        y = []
        y_imgs = []
        while xy.qsize() > 0:
            a,b,c,d = xy.get()
            x.append(a)
            selvecs.append(b)
            y.append(c)
            y_imgs.append(d)

        return [np.asarray(x), np.asarray(selvecs)], [np.asarray(y), np.asarray(y_imgs)]

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！

        return math.ceil(len(self.color_images_flat)/self.batch_size)


class D_DatageneratorS2():
    def __init__(self, color_path, batch_size, g_model, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.color_images = []
        for p in os.listdir(color_path):
            for f in os.listdir(color_path + p + '/'):
                self.color_images.append(color_path + p + '/' + f)
        random.shuffle(self.color_images)
        self.val = val
        self.valid_images = random.sample(self.color_images,batch_size*18)
        for d in self.valid_images:
            self.color_images.remove(d)
        self.g_model = g_model
        self.old_fake = {}

        # エラー回避のため、１回空振りさせる
        img_s = np.zeros((STANDARD_SIZE_S1[0],STANDARD_SIZE_S1[1],3))
        img = np.zeros((STANDARD_SIZE_S2[0],STANDARD_SIZE_S2[1]))
        with graph.as_default():
            g_model.predict_on_batch([np.array([img]), np.array([img_s]), np.array([0])])
        sleep(2)

    def __getitem__(self, idx):
        xy  = queue.Queue()
        # データの取得実装
        if self.val == 999:
            ytmp1 = self.valid_images
        else:
            ytmp1 = self.color_images[self.batch_size*idx:self.batch_size*(idx+1)]

        def func(path):
            img, line, img_small, selvec = image_arrange(path, resize=STANDARD_SIZE_S2)
            if random.randint(0,100) % 2 == 0:
                img_small = img.resize(STANDARD_SIZE_S1, Image.BICUBIC)
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5
            img_small = (np.asarray(img_small)-127.5)/127.5

            # rand = random.randint(0,100)    # BartchNormalization しない場合はこっち
            if self.val == 999:
                rand = random.randint(0,1000) 
            else:
                rand = self.val

            if rand%3 == 0:
                #本物
                x = img
                retvecs = selvec
                # lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([1.0 - r, 0.0 + r])

            elif rand%3 == 1:
                #本物だけど、色間違い
                x = img
                # カラーラベルランダム取得（正解以外）
                tmpcol = list(Colors.values())
                tmpcol.remove(selvec) 
                selvec2 = random.choice(tmpcol)
                retvecs = selvec2
                lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([0.0 + r, 1.0 - r])

            else:
                #偽物

                if random.randint(0,10) == 0 and path in self.old_fake:
                    #過去の生成画像を使用する
                    x = self.old_fake[path]
                else:
                    with graph.as_default():
                        ret = self.g_model.predict_on_batch([np.array([line]), np.array([img_small]), np.array([selvec])])
                    x = ret[0]

                retvecs = selvec
                # lines = line
                r = random.uniform(0.0, 0.2)
                y = np.asarray([0.0 + r, 1.0 - r])

                r = random.randint(0,self.batch_size*20)
                if r == 0:
                    filename = f'temp/gen2.png'
                    tmp = (x*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    filename = f'temp/gen2_in_line.png'
                    tmp = (line*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    filename = f'temp/gen2_in_gen1.png'
                    tmp = (img_small*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    filename = f'temp/gen2_true.png'
                    tmp = (img*127.5+127.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(tmp).save(filename,'png', optimize=True)

                    #偽物画像を保存しておく（次のDiscriminater用の学習へ）
                    self.old_fake[path] = x

                    if len(self.old_fake) > 1000:
                        del self.old_fake[random.choice(list(self.old_fake.keys() ))]

            xy.put([x, selvec, y])

        threads = []
        for path in ytmp1:
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        col = []
        y = []
        while xy.qsize() > 0:
            a,v,b = xy.get()
            x.append(a)
            col.append(v)
            y.append(b)

        return [np.asarray(x), np.asarray(col)], np.asarray(y)

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！
        sample_per_epoch = math.ceil(len(self.color_images)/self.batch_size)
        return sample_per_epoch


class Comb_DatageneratorS2():
    def __init__(self, color_path, batch_size, val=0):
        # コンストラクタ
        self.color_path = color_path
        self.batch_size = batch_size
        self.val = val
        self.color_images = {}
        self.color_images_flat = []
        self.valid_images = []
        for p in os.listdir(color_path):
            tmp = []
            for f in os.listdir(color_path + p + '/'):
                tmp.append(color_path + p + '/' + f)

            valid_tmp = random.sample(tmp, batch_size*2)
            self.valid_images.extend(valid_tmp)
            for d in valid_tmp:
                tmp.remove(d)
            random.shuffle(tmp)
            self.color_images[p] = tmp
            self.color_images_flat.extend(tmp)

        random.shuffle(self.color_images_flat)

    def __getitem__(self, idx):
        # データの取得実装
        if self.val == 999:
            ytmp1 = random.sample(self.valid_images, self.batch_size)
        else:
            ytmp1 = self.color_images_flat[self.batch_size*idx:self.batch_size*(idx+1)]
        xy  = queue.Queue()
        def func(path):
            img, line, img_small, selvec = image_arrange(path, resize=STANDARD_SIZE_S2)
            if random.randint(0,100) % 3 != 0:
                img_small = img.resize(STANDARD_SIZE_S1, Image.BICUBIC)
                img_small = img_small.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 0.1)) )
            line = (np.asarray(line)-127.5)/127.5
            img = (np.asarray(img)-127.5)/127.5
            img_small = (np.asarray(img_small)-127.5)/127.5

            r = random.uniform(0.0, 0.2)
            y = [1.0 - r, 0.0 + r]
            # y = [1.0, 0.0]

            xy.put([line, img_small, selvec, y, img])

        threads = []
        for path in ytmp1:
            #本物
            threads.append( threading.Thread(target=func, args=(path,) ))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        #順序がバラバラにならないように
        x = []
        s1gen = []
        col = []
        y = []
        y_imgs = []
        while xy.qsize() > 0:
            a,b,v,c,d = xy.get()
            x.append(a)
            s1gen.append(b)
            col.append(v)
            y.append(c)
            y_imgs.append(d)

        return [np.asarray(x), np.asarray(s1gen), np.asarray(col)], [np.asarray(y), np.asarray(y_imgs)]

    def __len__(self):
        # 全データ数をバッチサイズで割って、何バッチになるか返すよー！

        return math.ceil(len(self.color_images_flat)/self.batch_size)
