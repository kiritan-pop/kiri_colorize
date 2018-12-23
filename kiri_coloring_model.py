# coding: utf-8
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D,\
                        Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Reshape, Concatenate, Average, Reshape,\
                        GaussianNoise, LeakyReLU, BatchNormalization, Embedding

def build_discriminator():
    en_alpha=0.3
    stddev=0.15 #0.2でいいかな？

    input_label = Input(shape=(1,), name="d_s1_input_label")
    label = Embedding(input_dim=9,output_dim=3,input_length=1)(input_label)
    label = Reshape(target_shape=(1, 1, 3))(label)
    label = UpSampling2D(size=(128,128))(label) #128

    input_image = Input(shape=(128, 128, 3), name="d_s1_input_main")
    model = Concatenate()([input_image, label])
    model = GaussianNoise(stddev)(model)
    model = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(model) # >64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(model) # >32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=256,  kernel_size=4, strides=2, padding='same')(model) # >16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=256,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GlobalAveragePooling2D()(model)
    model = Dense(2)(model)
    truefake = Activation('softmax', name="d_s1_out1_trfk")(model)
    return Model(inputs=[input_image, input_label], outputs=[truefake])

def build_generator():
    en_alpha=0.3
    dec_alpha=0.1
    en_stddev=0.1  # 0でいいかな？
    de_stddev=0.1  # 0でいいかな？
    en_d_out=0.0   # エンコードのみにしたらどうだろう？
    de_d_out=0.0   # 
    input_label = Input(shape=(1,), name="g_s1_input_label")
    label = Embedding(input_dim=9,output_dim=3,input_length=1)(input_label)
    label = Reshape(target_shape=(1, 1, 3))(label)
    label = UpSampling2D(size=(128,128))(label) #128


    input_tensor = Input(shape=(128, 128), name="g_s1_input_main")
    model = Reshape(target_shape=(128, 128, 1))(input_tensor)
    model = Concatenate()([model,label])
    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e128 = model

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(model) # > 64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e64 = model

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=128,  kernel_size=4, strides=2, padding='same')(model) # 64-> 32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e32 = model

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=256, kernel_size=4, strides=2, padding='same')(model) # 32-> 16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e16 = model

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(model) # 16-> 8
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e8 = model

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e8])   # 順序はあまり影響しないかな
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same')(model) #8->16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e16])   # 順序はあまり影響しないかな
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(model)  #16->32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e32])
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(model)  #32->64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e64])
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(model)  #64->128
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=3  , kernel_size=3, strides=1, padding='same')(model)

    return Model(inputs=[input_tensor, input_label], outputs=[model])

def build_combined(generator, discriminator):
    return Model(inputs=[generator.inputs[0], generator.inputs[1]], outputs=[discriminator([generator.outputs[0], generator.inputs[1]]), generator.outputs[0]] )

def build_discriminatorS2():
    en_alpha=0.3
    stddev=0.15 #0.2でいいかな？

    input_label = Input(shape=(1,), name="d_s2_input_label")
    label = Embedding(input_dim=9,output_dim=3,input_length=1)(input_label)
    label = Reshape(target_shape=(1, 1, 3))(label)
    label = UpSampling2D(size=(512, 512))(label) #128

    input_image = Input(shape=(512, 512, 3), name="d_s2_input_main")
    model = Concatenate()([input_image, label])
    model = GaussianNoise(stddev)(model)
    model = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(model) # >64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(model) # >32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    model = Conv2D(filters=256,  kernel_size=4, strides=2, padding='same')(model) # >16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=256,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GlobalAveragePooling2D()(model)
    model = Dense(2)(model)
    truefake = Activation('softmax', name="d_s2_out1_trfk")(model)
    return Model(inputs=[input_image, input_label], outputs=[truefake])


def build_generatorS2():
    en_alpha=0.3
    dec_alpha=0.1
    en_stddev=0.1  # 0でいいかな？
    de_stddev=0.1  # 0でいいかな？
    en_d_out=0.0   # エンコードのみにしたらどうだろう？
    de_d_out=0.0   # 

    input_line = Input(shape=(512, 512), name="g_s2_input_line")
    line = Reshape(target_shape=(512, 512, 1))(input_line)

    input_color = Input(shape=(128, 128, 3), name="g_s2_input_color")
    color = UpSampling2D(size=(4, 4))(input_color) #128

    model = Concatenate()([line, color])
    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=32,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=64,  kernel_size=4, strides=2, padding='same')(model) # > 64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=64,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e64 = model

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=128,  kernel_size=4, strides=2, padding='same')(model) # 64-> 32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=128,  kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e32 = model

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=256, kernel_size=4, strides=2, padding='same')(model) # 32-> 16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e16 = model

    model = Dropout(en_d_out)(model)
    model = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(model) # 16-> 8
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)
    e8 = model

    model = GaussianNoise(en_stddev)(model)
    model = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e8])   # 順序はあまり影響しないかな
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same')(model) #8->16
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e16])   # 順序はあまり影響しないかな
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(model)  #16->32
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e32])
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(model)  #32->64
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Concatenate()([model,e64])
    model = Dropout(de_d_out)(model)
    model = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(model)  #64->128
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=dec_alpha)(model)

    model = GaussianNoise(de_stddev)(model)
    model = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(model)
    model = BatchNormalization(momentum=0.8)(model)
    model = LeakyReLU(alpha=en_alpha)(model)

    model = Conv2D(filters=3  , kernel_size=3, strides=1, padding='same')(model)

    return Model(inputs=[input_line, input_color], outputs=[model])

def build_combinedS2(generator, discriminator):
    return Model(inputs=[generator.inputs[0], generator.inputs[1], generator.inputs[2]], outputs=[discriminator([generator.outputs[0], generator.inputs[1]]), generator.outputs[0]] )

def build_frozen_discriminator(discriminator):
    frozen_d = Model(inputs=discriminator.inputs, outputs=discriminator.outputs)
    frozen_d.trainable = False
    return frozen_d