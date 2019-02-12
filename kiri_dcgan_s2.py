# coding: utf-8
import tensorflow as tf
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam,Adadelta,Nadam
from keras.utils import plot_model, multi_gpu_model
from keras.callbacks import EarlyStopping, LambdaCallback
from keras import backend
from keras.layers import Input, Dense, Dropout, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D,\
                        Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Reshape, Concatenate, Average, Reshape,\
                        GaussianNoise, LeakyReLU, BatchNormalization, Embedding, Flatten

import os,glob,sys,json,random,cv2,threading,queue,multiprocessing
from time import sleep
import numpy as np
import math
from PIL import Image, ImageOps, ImageFile, ImageChops, ImageFilter, ImageEnhance
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--queue_size", type=int, default=10)
    parser.add_argument("--worker", type=int, default=1)
    args = parser.parse_args()
    return args

def image_resize(img, resize):
    # アスペクト比維持
    tmp = img
    # tmp.thumbnail(resize,Image.BICUBIC)
    if tmp.mode == 'L':
        tmp = expand2square(tmp,(255,))
    else:
        tmp = expand2square(tmp,(255,255,255))
    return tmp.resize(resize,Image.BICUBIC)

def image_to_line(img): # img:RGBモード
    gray = img
    # 線画化
    gray = new_convert(gray, "L") #グレイスケール
    gray2 = gray.filter(ImageFilter.MaxFilter(random.choice([3,5])))
    senga_inv = ImageChops.difference(gray, gray2)
    senga_inv = ImageOps.invert(senga_inv)

    en = ImageEnhance.Contrast(senga_inv)
    senga_inv = en.enhance(random.choice([0.8,1.0,1.5,2.0,3.0]))
    en = ImageEnhance.Brightness(senga_inv)
    senga_inv = en.enhance(random.uniform(0.99,1.1))

    return senga_inv

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
    img = ImageOps.autocontrast(img)
    en = ImageEnhance.Color(img)
    img = en.enhance(random.choice([0.85, 1.0, 1.1, 1.25]))
    #画像加工ランダム値
    rotate_rate = random.randint(0,360) #回転
    mirror =  random.randint(0,100) % 2
    resize_rate_w = random.uniform(0.65,1.0)
    resize_rate_h = random.uniform(0.65,1.0)
    # 回転
    img = img.rotate(rotate_rate, expand=False, resample=Image.BICUBIC)

    r_img = img.resize( (int(img.width//resize_rate_w), int(img.height//resize_rate_h)), Image.BICUBIC)
    r_width = random.randint(0, r_img.width  - img.width)
    r_height = random.randint(0, r_img.height - img.height)
    img = r_img.crop((r_width, r_height, img.width+r_width, img.height+r_height))

    # 反転
    if mirror ==0:
        img = ImageOps.mirror(img)

    #線画化
    line = image_to_line(img)

    # リサイズ
    img = img.resize(resize, Image.BICUBIC)
    line = line.resize(resize, Image.BICUBIC)
    small_line = line.resize((128,128), Image.BICUBIC)

    # カラー情報(減色して、トップｎ個のカラー情報RGBを返す)
    qnum = 16
    p = img.resize((128,128), Image.BICUBIC).quantize(qnum)
    palette = p.getpalette()[:qnum*3]
    colors = sorted(p.getcolors(), key=lambda x: -x[0])[:8]
    rgb = []
    num = []
    for n, c in colors:
        rgb.append(palette[3*c:3*(c+1)])
        num.append(n)

    return img, small_line, line, rgb


def make_noise(num):
    RGB = np.random.uniform(-1.0,1.0,(num, 8, 3))
    RGB[:, :2, :] += 1.0  # 背景を白っぽくするため

    return RGB

class DataSet():
    def __init__(self, image_path, qsize=4096, valid=0.1, workers=1):
        images = []
        for p in os.listdir(image_path):
            for f in os.listdir(os.path.join(image_path, p)):
                images.append(os.path.join(image_path, p, f))
        random.shuffle(images)

        valid_images = random.sample(images, int(len(images)*valid))
        for d in valid_images:
            images.remove(d)

        self.image_que = multiprocessing.Queue(qsize)
        self.valid_image_que = multiprocessing.Queue(qsize)
        def datagen(que, img_path):
            while True:
                for path in img_path:
                    #本物
                    img, small_line, line_img, color = image_arrange(path, resize=(512,512))
                    img = (np.asarray(img)-127.5)/127.5
                    small_line = (np.asarray(small_line)-127.5)/127.5
                    line_img = (np.asarray(line_img)-127.5)/127.5
                    color = (np.asarray(color)-127.5)/127.5
                    que.put((img, small_line, line_img, color))

        len_images = len(images)
        for i in range(workers):
            multiprocessing.Process(target=datagen, args=(self.image_que, images[len_images*i//workers:len_images*(i+1)//workers])).start()
        multiprocessing.Process(target=datagen, args=(self.valid_image_que, valid_images)).start()

    def get_data(self, size, val=False):
        # データの取得実装
        if val:
            tmp_que = self.valid_image_que
        else:
            tmp_que = self.image_que

        images = np.zeros((size, 512, 512, 3))
        small_line_images = np.zeros((size, 128, 128))
        line_images = np.zeros((size, 512, 512))
        colors = np.zeros((size, 8, 3))
        for i in range(size):
            img, small_line, line, hist = tmp_que.get()
            images[i] = img
            small_line_images[i] = small_line
            line_images[i] = line
            colors[i] = hist

        return images, small_line_images, line_images, colors

class KiriDcgan():
    def __init__(self, image_path, batch_size, GPUs, g_s1_path, g_path, d_path, result_path='results/', qsize=4096, workers=1):
        self.dataset = DataSet(image_path, qsize=qsize, valid=0.1, workers=workers)
        self.g_path = g_path
        self.d_path = d_path
        self.batch_size = batch_size
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

        # ジェネレータ（Ｓ１）ロード
        if os.path.exists(g_s1_path):
            self.frozen_g_s1 = self.build_frozen_model(load_model(g_s1_path))
        else:
            raise ValueError("generator(S1) model path not found")

        # ジェネレータ（Ｓ２）ロードorビルド
        if os.path.exists(g_path):
            self.g_model_s2 = load_model(g_path)
        else:
            self.g_model_s2 = self.build_generator()

        def summary_write(line):
            f.write(line+"\n")
        with open('g_model_s2.txt', 'w') as f:
            self.g_model_s2.summary(print_fn=summary_write)
        plot_model(self.g_model_s2, to_file='g_model_s2.png')

        # マージジェネレータ（S1+S2）
        _tmpmdl = self.build_merged_generator(self.frozen_g_s1, self.g_model_s2)
        if GPUs > 1:
            self.merged_generator = multi_gpu_model(_tmpmdl, gpus=GPUs)
        else:
            self.merged_generator = _tmpmdl

        # 判定器（Ｓ２）ロードorビルド
        if os.path.exists(d_path):
            self.d_model_s2 = load_model(d_path)
        else:
            self.d_model_s2 = self.build_discriminator()

        if GPUs > 1:
            self.d_model_s2_tr = multi_gpu_model(self.d_model_s2, gpus=GPUs)
        else:
            self.d_model_s2_tr = self.d_model_s2
        self.d_model_s2_tr.compile(loss='categorical_crossentropy',   #binary_crossentropy
                        optimizer=Nadam(),
                        )
        with open('d_model_s1.txt', 'w') as f:
            self.d_model_s2.summary(print_fn=summary_write)

        self.frozen_d = self.build_frozen_model(self.d_model_s2)

        _tmpmdl = self.build_combined(self.frozen_g_s1, self.g_model_s2, self.frozen_d)
        if GPUs > 1:
            self.combined = multi_gpu_model(_tmpmdl, gpus=GPUs)
        else:
            self.combined = _tmpmdl

        self.combined.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                        loss_weights=[1.0, 1.0],
                        optimizer=Nadam(),
                        )

    def train(self, start_idx, total_epochs, save_epochs=1000, sub_epochs=100):
        for epoch in range(start_idx+1, total_epochs+1):
            # 100epoch毎にテスト画像生成、validuetion
            if epoch%sub_epochs == 0:
                self.g_on_epoch_end_sub(epoch)
                # validuate
                print(f'\tValidation :', end='')
                val_flg = True
            else:
                print(f'\repochs={epoch:6d}/{total_epochs:6d}:', end='')
                val_flg = False

            true_images, small_line_images, line_images, hists = self.dataset.get_data(self.batch_size, val=val_flg)
            y1 = np.zeros((self.batch_size, 2))
            for p in range(self.batch_size):
                r = random.uniform(0.0, 0.2)
                y1[p] = [1.0 - r, 0.0 + r]

            fake_images = self.merged_generator.predict_on_batch([small_line_images, hists, line_images])
            if epoch % 10 == 0:
                filename = os.path.join("temp/", f"gen.png")
                tmp = (fake_images[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)

                filename = os.path.join("temp/", f"line.png")
                tmp = (line_images[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)

                filename = os.path.join("temp/", f"true.png")
                tmp = (true_images[0]*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)

            y2 = np.zeros((self.batch_size, 2))
            for p in range(self.batch_size):
                r = random.uniform(0.0, 0.2)
                y2[p] = [0.0 + r, 1.0 - r]

            noise = make_noise(self.batch_size)
            y3 = np.zeros((self.batch_size, 2))
            for p in range(self.batch_size):
                r = random.uniform(0.0, 0.2)
                y3[p] =  [0.0 + r, 1.0 - r]

            d_loss_true = self.d_model_s2_tr.train_on_batch([true_images[:self.batch_size//3], hists[:self.batch_size//3]], y1[:self.batch_size//3])
            d_loss_fake = self.d_model_s2_tr.train_on_batch([fake_images[:self.batch_size//3], hists[:self.batch_size//3]], y2[:self.batch_size//3])
            d_loss_true_alt = self.d_model_s2_tr.train_on_batch([true_images[:self.batch_size//3], noise[:self.batch_size//3]], y3[:self.batch_size//3])

            d_loss = sum([d_loss_true, d_loss_fake, d_loss_true_alt])/3
            print(f'D_loss={d_loss:.3f}  ',end='')

            g_loss = self.combined.train_on_batch([small_line_images, hists, line_images], [y1, true_images])
            print(f'G_loss={g_loss[0]:.3f}({g_loss[1]:.3f},{g_loss[2]:.3f})',end='')

            # 100epoch毎に
            if epoch%sub_epochs == 0:
                print()

            # 1000epoch毎にモデルの保存、テスト実施
            if epoch%save_epochs == 0:
                self.on_epoch_end(epoch)


    def on_epoch_end(self,epoch):
        self.g_model_s2.save(self.g_path)
        self.d_model_s2.save(self.d_path)
        save_path = os.path.join(self.result_path, f'{epoch:06d}')
        os.makedirs(os.path.join(save_path,"d2"), exist_ok=True)
        os.makedirs(os.path.join(save_path,"g2"), exist_ok=True)
        self.d_model_s2.save(os.path.join(save_path, self.d_path))
        self.g_model_s2.save(os.path.join(save_path, self.g_path))

        self._discrimin_test(epoch, os.path.join(save_path,"d2"))
        self._generator_test(epoch, os.path.join(save_path,"g2"))

    def g_on_epoch_end_sub(self, epoch):
        self._generator_test(epoch, None, True)


    def _generator_test(self, epoch, result_path, short=False, num=3):
        if short:
            test_dir = './test_short/'
        else:
            test_dir = './test/'

        i_dirs = []
        for f in os.listdir(test_dir):
                i_dirs.append(test_dir + f)
        for image_path in random.sample(i_dirs,min([len(i_dirs),3])):
            exet = image_path.rsplit('.',1)[-1]
            tmp_name = image_path.rsplit('.',1)[0].rsplit('/',1)[-1]
            img = Image.open(image_path)
            img = new_convert(img, "L")
            # img = img.filter(ImageFilter.MinFilter(3))
            img512 = image_resize(img,(512,512))
            img512 = (np.asarray(img512)-127.5)/127.5
            img512 = img512.reshape((1,512,512))
            img512 = img512.repeat(num, axis=0)

            img = image_resize(img,(128,128))
            img = (np.asarray(img)-127.5)/127.5
            img = img.reshape((1,128,128))
            img = img.repeat(num, axis=0)
            noise = make_noise(num)
            results = self.merged_generator.predict_on_batch([img, noise, img512])
            for i, ret in enumerate(results):
                if short:
                    os.makedirs(f"temp2/s2{epoch:06}", exist_ok=True)
                    filename = f"temp2/s2{epoch:06}/{tmp_name}_{i:02}.{exet}"
                else:
                    filename = f"{result_path}/{tmp_name}_{i:02}.{exet}"

                tmp = (ret*127.5+127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(tmp).save(filename, optimize=True)


    def _discrimin_test(self, epoch, result_path, num=6):
        #判定
        true_imgs, small_line_images, line_images, hists = self.dataset.get_data(num)
        fake_imgs = self.merged_generator.predict_on_batch([small_line_images[-num:], hists, line_images[-num:]])
        imgs = np.concatenate([true_imgs,fake_imgs], axis=0)
        hists = hists.repeat(2, axis=0)
        results = self.d_model_s2_tr.predict_on_batch([imgs, hists])
        for i,(img,result) in enumerate(zip(imgs,results)):
            filename = f'd{i:02}[{result[0]:1.2f}][{result[1]:1.2f}].png'
            tmp = (img*127.5+127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(tmp).save(os.path.join(result_path, filename), 'png', optimize=True)


    def build_discriminator(self):
        en_alpha=0.3
        stddev=0.05

        input_color = Input(shape=(8,3), name="d_s1_input_color")
        color = Flatten()(input_color)
        color = Dense(32*32*8)(color)
        color = LeakyReLU(alpha=en_alpha)(color)
        color = Reshape(target_shape=(32, 32, 8))(color)

        input_image = Input(shape=(512, 512, 3), name="d_s1_input_main")

        model = GaussianNoise(stddev)(input_image)
        model = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        model = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(model) # >64
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        gap1  = GlobalAveragePooling2D()(model)
        model = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(model) # >32
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        gap2  = GlobalAveragePooling2D()(model)
        model = Conv2D(filters=256,  kernel_size=4, strides=2, padding='same')(model) # >16
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Conv2D(filters=256,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        gap3  = GlobalAveragePooling2D()(model)
        model = Conv2D(filters=512,  kernel_size=4, strides=2, padding='same')(model) # >8
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Concatenate()([model, color])
        model = Conv2D(filters=512,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        model = GlobalAveragePooling2D()(model)

        model = Concatenate()([gap1, gap2, gap3, model])
        model = Dense(2)(model)
        truefake = Activation('softmax', name="d_s1_out1_trfk")(model)
        return Model(inputs=[input_image, input_color], outputs=[truefake])

    def build_generator(self):
        en_alpha=0.3
        dec_alpha=0.1

        input_color_vec = Input(shape=(8,3), name="g_s1_input_color")
        color_vec = Flatten()(input_color_vec)
        color_vec = Dense(32*32*8)(color_vec)
        color_vec = LeakyReLU(alpha=en_alpha)(color_vec)
        color_vec = Reshape(target_shape=(32, 32, 8))(color_vec)

        input_color = Input(shape=(128, 128, 3), name="g_s2_input_color")
        color = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(input_color)
        color = BatchNormalization(momentum=0.8)(color)
        color = LeakyReLU(alpha=en_alpha)(color)

        color = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(color)
        color = BatchNormalization(momentum=0.8)(color)
        color = LeakyReLU(alpha=en_alpha)(color)

        color = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(color)
        color = BatchNormalization(momentum=0.8)(color)
        color = LeakyReLU(alpha=en_alpha)(color)

        color = Conv2D(filters=128,  kernel_size=4, strides=2, padding='same')(color)
        color = BatchNormalization(momentum=0.8)(color)
        color = LeakyReLU(alpha=en_alpha)(color)

        color = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(color)
        color = BatchNormalization(momentum=0.8)(color)
        color = LeakyReLU(alpha=en_alpha)(color)

        input_line = Input(shape=(512, 512), name="g_s2_input_line")
        model = Reshape(target_shape=(512, 512, 1))(input_line)
        model = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(model) # > 256
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        e64 = model

        model = Conv2D(filters=128,  kernel_size=4, strides=2, padding='same')(model) # 128
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        # model = Concatenate()([model, color])
        model = Conv2D(filters=256,  kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        e32 = model

        model = Conv2D(filters=256, kernel_size=4, strides=2, padding='same')(model) # 64
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        e16 = model

        model = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(model) # 32
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)
        e8 = model

        model = Dropout(0.5)(model)
        model = Concatenate()([model, color, color_vec])
        model = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = GaussianNoise(0.1)(model)
        model = Concatenate()([model,e8])   # 順序はあまり影響しないかな
        model = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same')(model) #16
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=dec_alpha)(model)

        model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Concatenate()([model,e16])   # 順序はあまり影響しないかな
        model = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(model)  #32
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=dec_alpha)(model)

        model = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Concatenate()([model,e32])
        model = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(model)  #32->64
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=dec_alpha)(model)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Concatenate()([model,e64])
        model = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(model)  #64->128
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=dec_alpha)(model)

        model = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=en_alpha)(model)

        model = Conv2D(filters=3  , kernel_size=3, strides=1, padding='same')(model)

        return Model(inputs=[input_line, input_color, input_color_vec], outputs=[model])


    def build_merged_generator(self, generator_s1, generator_s2):
        return Model(inputs=[generator_s1.inputs[0], generator_s1.inputs[1], generator_s2.inputs[0]], 
                    outputs=[generator_s2([generator_s2.inputs[0], generator_s1.outputs[0], generator_s1.inputs[1]])]
            )


    def build_combined(self, generator_s1, generator_s2, discriminator):
        merged_generator = self.build_merged_generator(generator_s1, generator_s2)
        return Model(inputs=[merged_generator.inputs[0], merged_generator.inputs[1], merged_generator.inputs[2]], 
            outputs=[discriminator([merged_generator.outputs[0],  merged_generator.inputs[1]]), merged_generator.outputs[0]])


    def build_frozen_model(self, model):
        frozen_d = Model(inputs=model.inputs, outputs=model.outputs)
        frozen_d.trainable = False
        return frozen_d


if __name__ == "__main__":
    #パラメータ取得
    args = get_args()
    #GPU設定
    config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=False, visible_device_list=args.gpu),
                allow_soft_placement=True, 
                log_device_placement=False
                )
    session = tf.Session(config=config)
    backend.set_session(session)
    GPUs = len(args.gpu.split(','))

    kiri = KiriDcgan(image_path='./class_col/', 
                    batch_size=args.batch_size, 
                    GPUs=GPUs, 
                    g_s1_path='g_model_s1.h5', 
                    g_path='g_model_s2.h5', 
                    d_path='d_model_s2.h5',
                    qsize=args.queue_size,
                    workers=args.worker)
    kiri.train(start_idx=args.idx, total_epochs=1000000, save_epochs=1000, sub_epochs=100)
