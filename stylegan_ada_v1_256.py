import tensorflow as tf
from keras import Model
from keras.layers import Input, \
    Conv2D, ReLU, \
    Activation, Conv2DTranspose, AveragePooling2D, MaxPooling2D, Multiply, DepthwiseConv2D, \
    Lambda, Dense, Flatten, Reshape, Layer, ZeroPadding2D, Add, Concatenate, LeakyReLU, UpSampling2D, add, concatenate
# from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
import keras.backend as K
# from keras import losses
# from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
import numpy as np
import os, time
from PIL import Image
from functools import reduce, partial
from LossHistory import LossHistory
from InstanceNormalization import InstanceNormalization


# from keras.utils import plot_model

# 禁用GPU
# if not os.path.exists("./use"):
#     os.environ['CUDA_VISIBLE_DEVICES'] = " "
#     print("已禁用GPU" + "=" * 20)


# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

class dataGenerator(object):

    # img [0,255]->[-1,1]

    def __init__(self, loc, im_size, data_path, flip=True):
        self.flip = flip
        # self.suffix = suffix
        self.files = []
        self.im_size = im_size
        self.now_index = 0
        self.test_index = 0
        self.n = 1e10
        self.test_num = 1000

        print("Importing Images...")

        try:
            os.mkdir("data/" + loc + "-npy-" + str(self.im_size))
        except:
            self.load_from_npy(loc)
            return

        for dirpath, dirnames, filenames in os.walk(data_path):
            # [f for f in filenames if f.endswith("." + str(self.suffix))]
            for filename in filenames:
                print('\r' + str(len(self.files)), end='\r')
                fname = os.path.join(dirpath, filename)
                temp = Image.open(fname).convert('RGB')
                temp = temp.resize((self.im_size, self.im_size), Image.BILINEAR)
                temp = np.array(temp, dtype='uint8')
                self.files.append(temp)
                if self.flip:
                    self.files.append(np.flip(temp, 1))

        self.files = np.array(self.files)
        np.save("data/" + loc + "-npy-" + str(im_size) + "/data.npy", self.files)

        self.n = self.files.shape[0]
        self.test = self.files[0:self.test_num]
        self.train = self.files[self.test_num:]
        np.random.seed(114514)
        np.random.shuffle(self.test)
        np.random.shuffle(self.train)

        print("Found " + str(self.n) + " images in " + loc + ".")

    def load_from_npy(self, loc):

        print("Loading from .npy files.")

        self.files = np.load("data/" + str(loc) + "-npy-" + str(self.im_size) + "/data.npy")

        self.n = self.files.shape[0]
        self.test = self.files[0:self.test_num]
        self.train = self.files[self.test_num:]
        np.random.seed(114514)
        np.random.shuffle(self.test)
        np.random.shuffle(self.train)
        print("Found " + str(self.n) + " images in " + loc + ".")

    def get_batch(self, num):
        out = []

        for i in range(num):
            out.append(self.train[self.now_index])
            self.now_index = (self.now_index + 1) % (self.n - self.test_num)

        return np.array(out).astype('float32') / 127.5 - 1

    def get_test_batch(self, num):
        out = []
        for i in range(num):
            out.append(self.test[self.test_index])
            self.test_index = (self.test_index + 1) % self.test_num

        return np.array(out).astype('float32') / 127.5 - 1


# 1,4,4,512
class Const(Layer):
    def __init__(self, **kwargs):
        super(Const, self).__init__(**kwargs)

    def build(self, input_shape):
        self.const = self.add_weight(name='const',
                                     shape=(1, 4, 4, 512),
                                     initializer='one',
                                     trainable=True)
        super(Const, self).build(input_shape)

    def call(self, x, **kwargs):
        return tf.tile(self.const, [tf.shape(x)[0], 1, 1, 1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 4, 512)


# w,no b
class AddNoise(Layer):
    def __init__(self, **kwargs):
        super(AddNoise, self).__init__(**kwargs)

    def build(self, input_shape):
        # 创建一个可训练的权重变量矩阵
        self.kernel = self.add_weight(name='weight',
                                      shape=(1, 1, input_shape[0][-1]),
                                      # 假设输入tensor只有一个维度（不算batch的维度）
                                      initializer='zero',
                                      trainable=True)  # 如果要定义可训练参数这里一定要选择True
        # self.bias = self.add_weight(name='bias',
        #                             shape=(1, 1, input_shape[0][-1]),
        #                             initializer='zero',
        #                             trainable=True)

        super(AddNoise, self).build(input_shape)  # 这行代码一定要加上，super主要是调用MyLayer的父类（Layer）的build方法

    def call(self, x, **kwargs):
        noise = x[1]
        _, w, h, c = x[1].shape
        # noise = tf.random.normal([tf.shape(x)[0], w, h, 1], dtype=x.dtype)
        return x[0] + noise * tf.reshape(tf.cast(self.kernel, x[0].dtype), [1, 1, 1, -1])
        # return x[0] + noise * tf.reshape(tf.cast(self.kernel, x[0].dtype), [1, 1, 1, -1]) + tf.reshape(
        #     tf.cast(self.bias, x[0].dtype), [1, 1, 1, -1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class PixelNor(Layer):
    def __init__(self, **kwargs):
        super(PixelNor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PixelNor, self).build(input_shape)

    def call(self, x, **kwargs):
        epsilon = 1e-8
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


class UpscaledConv2d(Conv2DTranspose):
    def __init__(self, **kwargs):
        kwargs['strides'] = 2
        super(UpscaledConv2d, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UpscaledConv2d, self).build(input_shape)

    def call(self, x, **kwargs):
        w = tf.transpose(self.kernel, [0, 1, 3, 2])  # [kernel, kernel, fmaps_out, fmaps_in]
        w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, self.filters]
        out = tf.nn.conv2d_transpose(x, w, os, strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
        # print(out.shape)
        return out


class Blur(Layer):
    def __init__(self, stride=1, **kwargs):
        super(Blur, self).__init__(**kwargs)
        self.stride = stride

    def build(self, input_shape):
        # Finalize filter kernel.
        f = [1, 2, 1]
        f = np.array(f, dtype=np.float32)
        if f.ndim == 1:
            f = f[:, np.newaxis] * f[np.newaxis, :]
        assert f.ndim == 2
        f /= np.sum(f)
        f = f[:, :, np.newaxis, np.newaxis]
        f = np.tile(f, [1, 1, int(input_shape[-1]), 1])
        self.f = tf.constant(f, dtype=tf.float32, name='filter')
        self.strides = [1, 1, self.stride, self.stride]
        super(Blur, self).build(input_shape)

    def call(self, x, **kwargs):
        # Convolve using depthwise_conv2d.
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
        x = tf.nn.depthwise_conv2d(x, self.f, strides=self.strides, padding='SAME', data_format='NHWC')
        x = tf.cast(x, orig_dtype)
        return x


class Augmentation(Layer):
    def __init__(self, p, axes=(1, 2), **kwargs):
        self.axes = tuple(axes)
        self.p = p
        if len(self.axes) != 2:
            raise ValueError("len(axes) must be 2.")
        super(Augmentation, self).__init__(**kwargs)

    def build(self, input_shape):
        tenor_shape = input_shape
        self.dim = len(tenor_shape)

        if self.axes[0] == self.axes[1] or np.absolute(self.axes[0] - self.axes[1]) == self.dim:
            raise ValueError("Axes must be different.")

        if (self.axes[0] >= self.dim or self.axes[0] < -self.dim
                or self.axes[1] >= self.dim or self.axes[1] < -self.dim):
            raise ValueError("Axes={} out of range for tensor of ndim={}."
                             .format(self.axes, self.dim))
        super(Augmentation, self).build(input_shape)

    def call(self, x, **kwargs):
        pp = tf.random.uniform(shape=(1,), minval=0, maxval=1)
        pred = pp > self.p.value()
        x = K.switch(
            condition=pred,
            then_expression=lambda: x,
            else_expression=lambda: self.rot(x)
        )
        return x

    def rot(self, tensor):
        k = tf.random.uniform(minval=0.3, maxval=0.8, shape=(3,))
        k1 = k[0]
        k2 = k[1]
        k3 = k[2]
        # print(int(k1 * k2 * 36), int(k1 * (1 - k2) * 36), int((1 - k1) * k3 * 36))
        ten_n = tf.shape(tensor)[0]
        ten_n = tf.cast(ten_n, tf.float32)
        ten1 = K.cast([k1 * k2 * ten_n, 0, 0, 0], tf.int32)
        ten2 = K.cast([k1 * (1 - k2) * ten_n, 0, 0, 0], tf.int32)
        ten3 = K.cast([(1 - k1) * k3 * ten_n, 0, 0, 0], tf.int32)
        ad = tf.constant([0, 256, 256, 3], tf.int32)

        tensor1 = tf.slice(tensor, [0, 0, 0, 0], ten1 + ad, name=None)

        img180 = tf.reverse(
            tf.reverse(tf.slice(tensor, ten1 + ten2 + ten3, tf.shape(tensor) - (ten1 + ten2 + ten3), name=None),
                       axis=[self.axes[0]]),
            axis=[self.axes[1]], name=None)

        axes_list = np.arange(0, self.dim)
        (axes_list[self.axes[0]], axes_list[self.axes[1]]) = (axes_list[self.axes[1]], axes_list[self.axes[0]])  # 替换

        img90 = tf.transpose(tf.reverse(tf.slice(tensor, ten1, ten2 + ad, name=None), axis=[self.axes[1]]),
                             perm=axes_list, name=None)
        img270 = tf.reverse(tf.transpose(tf.slice(tensor, ten1 + ten2, ten3 + ad, name=None), perm=axes_list),
                            axis=[self.axes[1]], name=None)

        return_ten = K.concatenate([tensor1, img90, img180, img270], axis=0)
        return return_ten

    def compute_output_shape(self, input_shape):
        return input_shape


class StyleGan:
    def __init__(self, batch_size, lr, bate_2, data_path, p, istrain=True):
        # 新建保存路径
        if not os.path.exists("./gan_images"):
            os.mkdir("./gan_images")
        # 新建模型保存路径
        if not os.path.exists("./StyleGan_MODEL/"):
            os.mkdir("./StyleGan_MODEL/")

        # 图片数据文件夹
        if os.path.exists("./use"):
            self.data_path = data_path
        else:
            self.data_path = data_path

        self.batch_size = batch_size
        self.lr = lr
        self.cha = 32

        self.noise = None  # 用于输出测试图片
        self.p = p
        self.Aug_p = K.variable(value=self.p)

        self.optimizer_G = Adam(self.lr[0], beta_1=0, beta_2=bate_2[0])
        # self.optimizer_G = Adam(self.lr, 0.5)
        # self.optimizer_G = RMSprop(self.lr)

        self.optimizer_D = Adam(self.lr[1], beta_1=0, beta_2=bate_2[1])
        # self.optimizer_D = Adam(self.lr, 0.5)
        # self.optimizer_D = RMSprop(self.lr)

        self.input_shape = (256, 256, 3)  # 原图输入
        self.input_shape_decoder = (512,)  # decoder输入

        self.G_mapping = self.build_G_mapping(self.input_shape_decoder)
        self.G_syn = self.build_G_syn(self.input_shape_decoder)
        self.G_syn2 = self.build_G_syn2(self.input_shape_decoder)
        self.D = self.build_D(self.input_shape)
        # self.G_mapping.summary()
        self.G_syn.summary()
        # self.D.summary()

        self.build_model_dis()
        self.build_model_gan()

        # self.model_disgan.summary()
        # self.model_gan.summary()

        if istrain:
            self.data = dataGenerator(self.data_path.split("\\")[-1], self.input_shape[0], self.data_path, flip=True)

        # if not os.path.exists("./use"):
        #     plot_model(self.G_mapping, to_file='./网络结构可视化/G_mapping.png', show_shapes=True)
        #     plot_model(self.G_syn, to_file='./网络结构可视化/G_syn.png', show_shapes=True)
        #     plot_model(self.D, to_file='./网络结构可视化/D.png', show_shapes=True)
        #     plot_model(self.model_disgan, to_file='./网络结构可视化/disgan.png', show_shapes=True)
        #     plot_model(self.model_gan, to_file='./网络结构可视化/gan.png', show_shapes=True)

    # ===========================网络部分=========================
    # 生成中级向量
    def build_G_mapping(self, input_shape=(512,)):
        inp = Input(shape=input_shape)
        x = inp
        for _ in range(8):
            x = Dense(input_shape[0])(x)
            x = ReLU()(x)
        return Model(inp, x, name="G_mapping")

    # 合成图像
    def build_G_syn(self, input_shape=(512,)):

        def AdaIn(inp, A):
            x = InstanceNormalization(axis=3, center=False, scale=False)(inp)
            c = inp.shape[3]
            Ax = Dense(c)(A)
            Bx = Dense(c)(A)
            x1 = Multiply()([x, Ax])
            x = Add()([x, x1])
            x = Add()([x, Bx])
            return x

        def block(inp, A, noises, fil, up=True):
            if up:
                out = UpSampling2D()(inp)
                # out = Conv2DTranspose(filters=fil, kernel_size=3, strides=2, padding='same')(inp)
                # out = LeakyReLU(0.2)(out)
                out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
            #                 out = Blur()(out)
            else:
                # out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(inp)
                # out = LeakyReLU(0.2)(out)
                out = inp
            out = AddNoise()([out, noises[0]])
            out = LeakyReLU(0.2)(out)
            #             out = PixelNor()(out)
            out = AdaIn(out, A)

            out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
            out = AddNoise()([out, noises[1]])
            out = LeakyReLU(0.2)(out)
            #             out = PixelNor()(out)
            out = AdaIn(out, A)
            return out

        def toRGB(inp, multiple=0):
            out = Conv2D(filters=3, kernel_size=1, padding='same',
                         kernel_initializer='he_normal')(inp)

            if multiple:
                for i in range(multiple):
                    out = UpSampling2D(interpolation='bilinear')(out)

            return out

        inp_w = Input(shape=input_shape)

        inp_noises = []
        for i in range(7):
            shape = (4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(Input(shape=shape))
            inp_noises.append(Input(shape=shape))

        inp_const = Const()(inp_w)

        x = block(inp_const, inp_w, inp_noises[0:2], 16 * self.cha, up=False)  # 4
        # out = toRGB(x, 1)

        x = block(x, inp_w, inp_noises[2:4], 16 * self.cha)  # 8
        # out = Add()([toRGB(x), out])
        # out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w, inp_noises[4:6], 16 * self.cha)  # 16
        # out = Add()([toRGB(x), out])
        # out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w, inp_noises[6:8], 16 * self.cha)  # 32
        out = toRGB(x, 1)

        x = block(x, inp_w, inp_noises[8:10], 8 * self.cha)  # 64
        out = Add()([toRGB(x), out])
        out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w, inp_noises[10:12], 4 * self.cha)  # 128
        out = Add()([toRGB(x), out])
        out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w, inp_noises[12:], 2 * self.cha)  # 256
        out = Add(name="before_tanh")([toRGB(x), out])

        # out = toRGB(x)
        out = Lambda(lambda x: x / 4)(out)
        out = Activation('tanh')(out)

        # input 13个
        return Model([inp_w] + inp_noises, out, name="G_synthesis")

    def build_G_syn2(self, input_shape=(512,)):

        def AdaIn(inp, A):
            x = InstanceNormalization(axis=3, center=False, scale=False)(inp)
            c = inp.shape[3]
            Ax = Dense(c)(A)
            Bx = Dense(c)(A)
            x1 = Multiply()([x, Ax])
            x = Add()([x, x1])
            x = Add()([x, Bx])
            return x

        def block(inp, A, noises, fil, up=True):
            if up:
                out = UpSampling2D()(inp)
                # out = Conv2DTranspose(filters=fil, kernel_size=3, strides=2, padding='same')(inp)
                # out = LeakyReLU(0.2)(out)
                out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
            #                 out = Blur()(out)
            else:
                # out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(inp)
                # out = LeakyReLU(0.2)(out)
                out = inp
            out = AddNoise()([out, noises[0]])
            out = LeakyReLU(0.2)(out)
            #             out = PixelNor()(out)
            out = AdaIn(out, A)

            out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
            out = AddNoise()([out, noises[1]])
            out = LeakyReLU(0.2)(out)
            #             out = PixelNor()(out)
            out = AdaIn(out, A)
            return out

        def toRGB(inp, multiple=0):
            out = Conv2D(filters=3, kernel_size=1, padding='same',
                         kernel_initializer='he_normal')(inp)

            if multiple:
                for i in range(multiple):
                    out = UpSampling2D(interpolation='bilinear')(out)

            return out

        inp_w = Input(shape=input_shape)
        inp_w2 = Input(shape=input_shape)
        inp_w3 = Input(shape=input_shape)
        inp_w4 = Input(shape=input_shape)

        inp_noises = []
        for i in range(7):
            shape = (4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(Input(shape=shape))
            inp_noises.append(Input(shape=shape))

        inp_const = Const()(inp_w)

        x = block(inp_const, inp_w, inp_noises[0:2], 16 * self.cha, up=False)  # 4
        # out = toRGB(x, 1)

        x = block(x, inp_w, inp_noises[2:4], 16 * self.cha)  # 8
        # out = Add()([toRGB(x), out])
        # out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w, inp_noises[4:6], 16 * self.cha)  # 16
        # out = Add()([toRGB(x), out])
        # out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w, inp_noises[6:8], 16 * self.cha)  # 32
        out = toRGB(x, 1)

        x = block(x, inp_w2, inp_noises[8:10], 8 * self.cha)  # 64
        out = Add()([toRGB(x), out])
        out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w3, inp_noises[10:12], 4 * self.cha)  # 128
        out = Add()([toRGB(x), out])
        out = UpSampling2D(interpolation='bilinear')(out)

        x = block(x, inp_w4, inp_noises[12:], 2 * self.cha)  # 256
        out = Add(name="before_tanh")([toRGB(x), out])

        # out = toRGB(x)
        out = Lambda(lambda x: x / 4)(out)
        out = Activation('tanh')(out)

        # input 13个
        return Model([inp_w, inp_w2, inp_w3, inp_w4] + inp_noises, out, name="G_synthesis")

    # 判别器
    def build_D(self, input_shape):

        def fRGB(x, img, fil):
            img = Conv2D(filters=fil, kernel_size=1, padding='same',
                         kernel_initializer='he_normal')(img)
            if x is not None:
                img = Add()([x, img])
            return img

        def block(inp, fil, p=True):

            out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(inp)
            out = LeakyReLU(0.2)(out)

            if p:
                strides = 2
                skip = Conv2D(filters=fil, kernel_size=1, padding='same', strides=strides,
                              kernel_initializer='he_normal')(
                    inp)
            else:
                strides = 1
            out = Conv2D(filters=fil, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal')(
                out)

            if p:
                out = Lambda(lambda x: (x[0] + x[1]) * (1 / np.sqrt(2)))([skip, out])
            out = LeakyReLU(0.2)(out)

            return out

        inp = Input(shape=input_shape)
        y = inp
        # x = None

        # x = fRGB(x, y, 2 * self.cha)
        x = block(y, 2 * self.cha)  # 128
        # y = AveragePooling2D()(y)

        # x = fRGB(x, y, 2 * self.cha)
        x = block(x, 4 * self.cha)  # 64
        # y = AveragePooling2D()(y)

        # x = fRGB(x, y, 4 * self.cha)
        x = block(x, 8 * self.cha)  # 32
        # y = AveragePooling2D()(y)

        # x = fRGB(x, y, 8 * self.cha)
        x = block(x, 16 * self.cha)  # 16
        # y = AveragePooling2D()(y)

        # x = fRGB(x, y, 16 * self.cha)
        x = block(x, 16 * self.cha)  # 8
        # y = AveragePooling2D()(y)

        # x = fRGB(x, y, 16 * self.cha)
        x = block(x, 16 * self.cha)  # 4
        # y = AveragePooling2D()(y)

        # x = fRGB(x, y, 16 * self.cha)
        x = Conv2D(filters=16 * self.cha, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = LeakyReLU(0.2)(x)
        # x = block(x, 16 * self.cha, p=False)  # 4

        x = Flatten()(x)
        x = Dense(16 * self.cha)(x)
        x = Dense(1)(x)

        return Model(inp, x, name="Dis")

    # 训练判别器的模型
    def build_model_dis(self):
        self.G_mapping.trainable = False
        for layer in self.G_mapping.layers:
            layer.trainable = False

        self.G_syn.trainable = False
        for layer in self.G_syn.layers:
            layer.trainable = False

        self.D.trainable = True
        for layer in self.D.layers:
            layer.trainable = True

        o_img = Input(shape=self.input_shape, name="o_img_input")
        mapping_noise_input = Input(shape=self.input_shape_decoder, name="noise_input")
        inp_noises = []
        for i in range(7):
            shape = (4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(Input(shape=shape))
            inp_noises.append(Input(shape=shape))

        # 得到中间向量
        W = self.G_mapping(mapping_noise_input)

        # 生成图像
        img_g = self.G_syn([W] + inp_noises)

        img_g_aug = Augmentation(p=self.Aug_p)(img_g)
        o_img_aug = Augmentation(p=self.Aug_p)(o_img)

        # 判别器
        fake_grade = self.D(img_g_aug)
        real_grade = self.D(o_img_aug)

        def Loss(grade):
            loss = K.log(K.exp(grade[0]) + 1)
            loss += K.log(K.exp(-grade[1]) + 1)
            # loss = K.relu(1 + grade[0])
            # loss += K.relu(1 - grade[1])
            return loss

        loss = Lambda(Loss, name="Loss")([fake_grade, real_grade])

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=o_img, weight=10)

        self.model_disgan = Model([o_img, mapping_noise_input] + inp_noises, [loss, real_grade], name="disgan")

        self.model_disgan.compile(loss=[lambda y_t, y_p: y_p, partial_gp_loss],
                                  optimizer=self.optimizer_D, )

    # 训练生成器模型
    def build_model_gan(self):
        self.G_mapping.trainable = True
        for layer in self.G_mapping.layers:
            layer.trainable = True

        self.G_syn.trainable = True
        for layer in self.G_syn.layers:
            layer.trainable = True

        self.D.trainable = False
        for layer in self.D.layers:
            layer.trainable = False

        mapping_noise_input = Input(shape=self.input_shape_decoder, name="noise_input")
        inp_noises = []
        for i in range(7):
            shape = (4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(Input(shape=shape))
            inp_noises.append(Input(shape=shape))

        # 得到中间向量
        W = self.G_mapping(mapping_noise_input)
        # 生成图像
        img_g = self.G_syn([W] + inp_noises)

        img_g = Augmentation(p=self.Aug_p)(img_g)

        # 判别器
        fake_grade = self.D(img_g)

        def Loss(grade):
            loss = K.log(K.exp(-grade) + 1)
            # loss = -grade
            return loss

        loss = Lambda(Loss, name="Loss")(fake_grade)

        self.model_gan = Model([mapping_noise_input] + inp_noises, loss, name="gan")

        self.model_gan.compile(loss=[lambda y_t, y_p: y_p],
                               optimizer=self.optimizer_G, )

    # gp-loss
    def gradient_penalty_loss(self, y_true, y_pred, sample_weight, averaged_samples, weight=10):
        gradients = K.gradients(y_pred, averaged_samples)
        gradients_sqr = K.square(gradients[0])
        gradient_penalty = K.sum(gradients_sqr,
                                 axis=np.arange(1, len(gradients_sqr.shape)))

        # (weight / 2) * (||grad||^2 - 1)^2
        # Penalize the gradient norm
        gradient_penalty = K.square(K.sqrt(gradient_penalty) - 1)

        return K.mean(gradient_penalty) * (weight / 2)

    # ===================训练===================

    def create_noise(self, loc, scale, shape, threshold=2.0):
        noise = np.random.normal(loc, scale, shape)
        cut = np.array(noise < -threshold) + np.array(noise > threshold)
        noise[cut] = np.random.random(len(noise[cut])) * 2 * threshold - threshold
        return noise

    def change_ada_p(self):
        train_img = self.data.get_batch(self.batch_size * 2)
        test_img = self.data.get_test_batch(self.batch_size * 2)
        g_img = self.predict_Gan(self.batch_size * 2)

        train = self.D.predict(train_img)
        test = self.D.predict(test_img)
        fake = self.D.predict(g_img)

        r_t = np.mean(np.sign(train))
        train = np.mean(train)
        test = np.mean(test)
        fake = np.mean(fake)
        r_v = abs(train - test) / abs(train - fake)
        if r_t > 0:
            if r_v > 0.2:
                self.p += 0.002
                self.p = self.p if self.p < 0.8 else 0.8
            else:
                self.p -= 0.001
                self.p = self.p if self.p > 0 else 0
            K.set_value(self.Aug_p, self.p)

        print("[r_v: {0:.4}, r_t: {1:.4}, p: {2:.4}]".format(r_v, r_t, float(self.p)))
        print("[means train: {0:.4}, test: {1:.4}, fake: {2:.4}]\n".format(train, test, fake))
        return r_v, r_t, self.p

    def train(self, Epochs, save_interval):
        ones = np.ones((self.batch_size, 1), dtype=np.float32)
        # nones = -ones
        # zeros = np.zeros((self.batch_size, 1), dtype=np.float32)

        save_interval_up = True
        r_v, r_t, p = 0, 0, 0

        losshistroy = LossHistory()
        losshistroy.on_train_begin(['d_loss1', 'd_loss2', 'g_loss', 'r_v', 'r_t', 'p'])
        self.save_imgs(0)

        for epoch in range(1, Epochs + 1):
            s_t = time.time()
            # =====================训练判别模型======================
            d_loss = [6, 6]
            for o in range(1):
                imgs = self.data.get_batch(self.batch_size)  # 选出数据
                noise = self.create_noise(0.0, 1.0, (self.batch_size, self.input_shape_decoder[0]))
                inp_noises = []
                for i in range(7):
                    shape = (self.batch_size, 4 * 2 ** i, 4 * 2 ** i, 1)
                    inp_noises.append(self.create_noise(0.0, 1.0, shape))
                    inp_noises.append(self.create_noise(0.0, 1.0, shape))
                # 训练并计算loss
                d_loss = self.model_disgan.train_on_batch([imgs, noise] + inp_noises, [ones, ones])
                # print('(%d) [D loss1: %f,loss2: %f]' % (o, d_loss[1], d_loss[2]))
            # ========================调整aug_p=====================
            if epoch % 5 == 0:
                r_v, r_t, p = self.change_ada_p()
            # ========================训练生成模型====================
            # imgs = self.data.get_batch(self.batch_size)  # 选出数据
            noise = self.create_noise(0.0, 1.0, (self.batch_size, self.input_shape_decoder[0]))
            inp_noises = []
            for i in range(7):
                shape = (self.batch_size, 4 * 2 ** i, 4 * 2 ** i, 1)
                inp_noises.append(self.create_noise(0.0, 1.0, shape))
                inp_noises.append(self.create_noise(0.0, 1.0, shape))
            g_loss = self.model_gan.train_on_batch([noise] + inp_noises, ones)
            # ========================================================
            e_t = time.time()
            print(
                "\n\n(%d/%d) t:%fs \n[D loss1: %f,loss2: %f]\n[G loss: %f]" % (
                    epoch, Epochs, e_t - s_t, d_loss[1], d_loss[2], g_loss))
            losshistroy.on_epoch_end(
                {"d_loss1": d_loss[1], "d_loss2": d_loss[2],
                 "g_loss": g_loss, 'r_v': r_v, 'r_t': r_t, 'p': p})
            # 输出图片,保存loss
            if epoch % save_interval[0] == 0:
                if save_interval_up and epoch > 3000:
                    save_interval_up = False
                    save_interval[0] *= 2
                self.save_imgs(epoch)
                losshistroy.loss_plot(epoch)
                losshistroy.save_to_file()
            # 保存模型
            if epoch % save_interval[1] == 0:
                self.save_model(epoch % (save_interval[1] * 8))

        # 训练结束
        self.save_model(-1)
        self.save_imgs(-1)
        # ae_img_test = self.data.get_batch(18)
        # self.predict_ae(ae_img_test)
        losshistroy.loss_plot(-1)
        losshistroy.save_to_file()

    # ==================保存===================

    def save_model(self, epoch):
        save_dir = "./StyleGan_MODEL/"
        self.G_mapping.save_weights(save_dir + 'G_mapping_{0}.h5'.format(epoch))
        self.G_syn.save_weights(save_dir + 'G_syn_{0}.h5'.format(epoch))
        self.D.save_weights(save_dir + 'D_{0}.h5'.format(epoch))

    def load_model(self, V):
        if V:
            save_dir = "./StyleGan_MODEL/v" + str(V) + "/"
            print("加载 v" + str(V))
        else:
            save_dir = "./StyleGan_MODEL/FID 54(G后4层相加32开始/FID 34/"
        v_epoch = 1000
        self.G_mapping.load_weights(save_dir + 'G_mapping_{0}.h5'.format(v_epoch))
        self.G_syn.load_weights(save_dir + 'G_syn_{0}.h5'.format(v_epoch), by_name=True)
        self.G_syn2.load_weights(save_dir + 'G_syn_{0}.h5'.format(v_epoch))
        self.D.load_weights(save_dir + 'D_{0}.h5'.format(v_epoch))
        # for i in range(1, 13):
        #     layer = self.G_syn.get_layer("add_noise_" + str(i))
        #     layer.set_weights(layer.get_weights() + 1)

        print("加载完成")

    # ==================测试===================

    def get_w_avg(self):
        try:
            return np.load(r'./StyleGan_MODEL/w_avg.npy')
        except:
            n = 10000
            noise = self.create_noise(0.0, 1.0, (n, self.input_shape_decoder[0]))
            w = self.G_mapping.predict(noise)
            w_avg = np.mean(w, axis=0)
            np.save(r'./StyleGan_MODEL/w_avg.npy', w_avg)
            return w_avg

    def create_image(self, img_array, r, c, shape, w):
        img_out = np.zeros((r * shape + (r - 1) * w, c * shape + (c - 1) * w, 3))

        cnt = 0
        for i in range(r):
            for j in range(c):
                ws = i * shape + i * w
                we = (i + 1) * shape + i * w
                hs = j * shape + j * w
                he = (j + 1) * shape + j * w
                try:
                    img_out[ws:we, hs:he, :] = img_array[cnt, :, :, :]
                    cnt += 1
                except:
                    cnt += 1
                    continue

        img_out *= 255
        img_out = img_out.astype("int32")
        # img_out = np.clip(img_out, 0, 255)
        img = Image.fromarray(np.uint8(img_out))
        return img

    def save_imgs(self, epoch, r=6, c=6):
        if self.noise is None:
            self.noise = self.create_noise(0.0, 1.0, (r * c, self.input_shape_decoder[0]))
            noise = self.noise
            self.inp_noises = []
            for i in range(7):
                shape = (r * c, 4 * 2 ** i, 4 * 2 ** i, 1)
                self.inp_noises.append(self.create_noise(0.0, 1.0, shape))
                self.inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises = self.inp_noises
        else:
            self.noise = self.create_noise(0.0, 1.0, (r * c, self.input_shape_decoder[0]))
            noise = self.noise
            self.inp_noises = []
            for i in range(7):
                shape = (r * c, 4 * 2 ** i, 4 * 2 ** i, 1)
                self.inp_noises.append(self.create_noise(0.0, 1.0, shape))
                self.inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises = self.inp_noises
            # inp_noises[0] = self.create_noise(0.0, 1.0, (r * c, 4, 4, 1))

        W = self.G_mapping.predict(noise)
        w_avg = self.get_w_avg()
        W = w_avg + 0.65 * (W - w_avg)
        gen_imgs = self.G_syn.predict([W] + inp_noises)
        gen_imgs = 0.5 * gen_imgs + 0.5

        shape = self.input_shape[0]
        w = 3
        img = self.create_image(gen_imgs, r, c, shape, w)

        img.save("./gan_images/face_{0}.png".format(str(epoch)))

    def predict_discriminator(self, index):
        batch_size = 128

        noise = self.create_noise(0.0, 1.0, (batch_size, self.input_shape_decoder[0]))
        inp_noises = []
        for i in range(7):
            shape = (batch_size, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises.append(self.create_noise(0.0, 1.0, shape))

        W = self.G_mapping.predict(noise)
        gen_imgs = self.G_syn.predict([W] + inp_noises)

        imgs = self.data.get_batch(batch_size)
        truth = self.D.predict(imgs)
        fake = self.D.predict(gen_imgs)

        imgs = []
        f_list = []
        for dirpath, dirnames, filenames in os.walk(r"E:\p站图片\database\洗出来的"):
            # [f for f in filenames if f.endswith("." + str(self.suffix))]
            for filename in filenames:
                fname = os.path.join(dirpath, filename)
                f_list.append(fname)
        np.random.shuffle(f_list)

        for i in range(batch_size):
            fname = f_list[i]
            temp = Image.open(fname).convert('RGB')
            temp = temp.resize((256, 256), Image.BILINEAR)
            temp = np.array(temp, dtype='uint8')
            imgs.append(temp)

        imgs = np.array(imgs)
        truth2 = self.D.predict(imgs)

        # r, c = 4, 10
        # fig, axs = plt.subplots(r, c)
        #
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i, j].imshow(imgs[cnt, :, :, :] if i < 2 else gan_imgs[cnt, :, :, :])
        #         axs[i, j].axis('off')
        #         cnt += 1
        #         if cnt == batch_size:
        #             cnt = 0
        # plt.show()
        # plt.close()

        plt.plot([i for i in range(batch_size)], truth, label="truth")
        plt.plot([i for i in range(batch_size)], truth2, label="truth2")
        plt.plot([i for i in range(batch_size)], fake, label="fake")
        plt.grid(True)
        plt.savefig("./gan_images/predict_{0}.png".format(index))
        plt.ylabel('loss')
        plt.legend(loc="upper left")
        plt.show()
        plt.close()

    def predict_Gan(self, num):
        batch_size = num

        noise = self.create_noise(0.0, 1.0, (batch_size, self.input_shape_decoder[0]))
        inp_noises = []
        for i in range(7):
            shape = (batch_size, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises.append(self.create_noise(0.0, 1.0, shape))

        W = self.G_mapping.predict(noise)
        w_avg = self.get_w_avg()
        W = w_avg + 0.5 * (W - w_avg)
        gan_imgs = self.G_syn.predict([W] + inp_noises)

        return gan_imgs

    def predict_change(self, num, move_mode="both"):
        # move_mode 属于('both','A','B')
        # both:ABnoise都渐变
        # A:Anoise渐变
        # B:Bnoise渐变

        points = 30
        middle = 10

        if move_mode in ('both', 'A'):
            # 渐变Anoise
            noise = []
            noise1 = self.create_noise(0.0, 1.0, (self.input_shape_decoder[0],))
            noise2 = self.create_noise(0.0, 1.0, (self.input_shape_decoder[0],))
            for j in range(points):
                vector = (noise2 - noise1) / middle

                for i in range(1, middle + 1):
                    n = noise1 + vector * i
                    noise.append(n)

                noise1 = noise2
                noise2 = self.create_noise(0.0, 1.0, (self.input_shape_decoder[0],))
            noise = np.array(noise)
        else:
            # 固定Anoise
            noise = np.tile(self.create_noise(0.0, 1.0, (1, self.input_shape_decoder[0])),
                            (points * middle, 1))

        inp_noises = []
        for i in range(7):
            shape = (points * middle, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises.append(self.create_noise(0.0, 1.0, shape))

        if move_mode in ('both', 'B'):
            # 渐变Bnoise
            first_inp_noise = []
            inp_noises1 = self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1)
            inp_noises2 = self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1)
            for j in range(points):
                vector = (inp_noises2 - inp_noises1) / (middle - 1)
                for i in range(0, middle):
                    n = inp_noises1 + vector * i
                    first_inp_noise.append(n)
                inp_noises1 = inp_noises2
                inp_noises2 = self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1.5)
            first_inp_noise = np.array(first_inp_noise)
            inp_noises[0] = first_inp_noise
        else:
            # 固定Bnoise
            inp_noises[0] = np.tile(self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1), (points * middle, 1, 1, 1))

        W = self.G_mapping.predict(noise)
        w_avg = self.get_w_avg()
        W = w_avg + 0.4 * (W - w_avg)
        gen_imgs = self.G_syn.predict([W] + inp_noises)
        gen_imgs = 0.5 * gen_imgs + 0.5

        img = self.create_image(gen_imgs, points, middle, self.input_shape[0], 3)
        img.save(r'F:\乱搞\BiGan\gan_images\gif/all_{0}.png'.format(num))

        for i in range(points * middle):
            img_out = gen_imgs[i, :]
            img_out *= 255
            img_out = img_out.astype("int32")
            img = Image.fromarray(np.uint8(img_out))
            img.save(r'F:\乱搞\BiGan\gan_images\gif/{0}.png'.format(i))

        from makegif import get_gif
        pics_dir = r'F:\乱搞\BiGan\gan_images\gif/gif_{0}'.format(num)
        save_name = get_gif(pics_dir, 200)
        print('制作完成。所属文件:{}'.format(save_name))

    def change_W(self, num):
        points = 64
        middle = 6

        # 渐变Anoise
        noise = []
        noise.append(self.create_noise(0.0, 1.0, (self.input_shape_decoder[0],)))
        noise = np.array(noise)

        inp_noises = []
        for i in range(7):
            shape = (points * middle + 2, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
        inp_noises[0] = np.tile(self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1), (points * middle + 2, 1, 1, 1))

        W_n = self.G_mapping.predict(noise)
        w_avg = self.get_w_avg()

        W = []
        vector = np.ones((self.input_shape_decoder[0],), dtype=np.float) * 0.5
        for i in range(points):
            mask = np.zeros((self.input_shape_decoder[0],), dtype=np.float)
            mask[i] = 1.0
            for j in range(-3, middle - 3):
                n = w_avg[0] + vector * mask * float(j)
                W.append(n)
        W.append(W_n[0])
        W.append(w_avg)
        W = np.array(W)
        print(W.shape)

        # W = w_avg + 0.7 * (W - w_avg)
        gen_imgs = self.G_syn.predict([W] + inp_noises)
        gen_imgs = 0.5 * gen_imgs + 0.5

        img = self.create_image(gen_imgs, points, middle, self.input_shape[0], 3)
        img.save(r'F:\乱搞\BiGan\gan_images\gif/changeW_{0}.png'.format(num))

        img = self.create_image(gen_imgs[-2:], 1, 2, self.input_shape[0], 3)
        img.save(r'F:\乱搞\BiGan\gan_images\gif/changeWavg_{0}.png'.format(num))

    def G_mapping_avg(self):
        n = 100000
        noise = self.create_noise(0.0, 1.0, (n, self.input_shape_decoder[0]))
        w = self.G_mapping.predict(noise)
        w_avg = np.mean(w, axis=0)
        print(w_avg.shape)

        np.save(r'./StyleGan_MODEL/w_avg.npy', w_avg)
        np.save(r'./StyleGan_MODEL/w100000.npy', w)

        inp_noises = []
        for i in range(7):
            shape = (64, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
        img = self.G_syn.predict([np.tile(w_avg, (64, 1))] + inp_noises)
        img = img * 0.5 + 0.5
        img = self.create_image(img, 8, 8, self.input_shape[0], 3)
        img.save("./gan_images/face_avgw.png")

    def K_means_w(self):
        k = 71
        w = np.load(r'./StyleGan_MODEL/w_mid{}.npy'.format(k))
        inp_noises = []
        for i in range(7):
            shape = (k + 1, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
            inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
        inp_noises[0] = np.tile(self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1), (k + 1, 1, 1, 1))

        w_avg = self.get_w_avg()
        w = w_avg + 0.7 * (w - w_avg)
        w = np.concatenate([w, np.array([w_avg])], axis=0)

        img = self.G_syn.predict([w] + inp_noises)
        img = img * 0.5 + 0.5
        img = self.create_image(img, 9, 8, self.input_shape[0], 3)
        img.save("./gan_images/face_kmeans_w{}.png".format(k))

    def predict_2w(self):
        # 测试两个w
        k = 71
        w = np.load(r'./StyleGan_MODEL/w_mid{}.npy'.format(k))[[8, -2]]
        inp_noises = []
        for i in range(7):
            shape = (5, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
            inp_noises.append(self.create_noise(0.0, 1.0, shape, threshold=1))
        inp_noises[0] = np.tile(self.create_noise(0.0, 1.0, (4, 4, 1), threshold=1), (5, 1, 1, 1))

        w_avg = self.get_w_avg()
        w = w_avg + 0.7 * (w - w_avg)

        w1 = []
        vector = (w[1] - w[0]) / 4

        for i in range(0, 5):
            n = w[0] + vector * i
            w1.append(n)
        w1 = np.array(w1)
        img = self.G_syn.predict([w1] + inp_noises)
        img = img * 0.5 + 0.5

        img2 = self.G_syn2.predict(
            [np.array([w[0], w[1], w[1], w[1], w[1]]), np.array([w[0], w[0], w[1], w[1], w[1]]),
             np.array([w[0], w[0], w[0], w[1], w[1]]), np.array([w[0], w[0], w[0], w[0], w[1]])] + inp_noises)
        img2 = img2 * 0.5 + 0.5

        img = np.concatenate([img2, img], axis=0)

        img = self.create_image(img, 2, 5, self.input_shape[0], 3)
        img.show()
        # img.save("./gan_images/face_kmeans_w{}.png".format(k))

    def middle_output(self, num):
        layers_name = ['conv2d_8', 'conv2d_11', 'conv2d_14', 'conv2d_17', 'before_tanh']
        g_layers = []
        for l in layers_name:
            g_layers.append(self.G_syn.get_layer(l).output)

        model_g = Model(input=self.G_syn.inputs, output=g_layers)
        noise = self.create_noise(0.0, 1.0, (1, self.input_shape_decoder[0]))
        inp_noises = []
        for i in range(7):
            shape = (1, 4 * 2 ** i, 4 * 2 ** i, 1)
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
            inp_noises.append(self.create_noise(0.0, 1.0, shape))
        W = self.G_mapping.predict(noise)
        w_avg = self.get_w_avg()
        W = w_avg + 0.65 * (W - w_avg)
        gan_imgs = model_g.predict([W] + inp_noises)

        r, c = 1, 5
        shape = self.input_shape[0]
        w = 3
        img_out = np.ones((r * shape + (r - 1) * w, (c + 1) * shape + c * w, 3)) * 0.5

        cnt = 0
        for i in range(r):
            for j in range(c):
                gi = 0.5 * gan_imgs[cnt][0, :, :, :] + 0.5
                img = Image.fromarray(np.uint8(gi))
                img = img.resize((shape, shape), Image.BOX)

                ws = i * shape + i * w
                we = (i + 1) * shape + i * w
                hs = j * shape + j * w
                he = (j + 1) * shape + j * w
                img_out[ws:we, hs:he, :] = np.array(img)
                cnt += 1

        gan_imgs = self.G_syn.predict([W] + inp_noises)
        img = 0.5 * gan_imgs[0, :] + 0.5

        r -= 1
        ws = r * shape + r * w
        we = (r + 1) * shape + r * w
        hs = c * shape + c * w
        he = (c + 1) * shape + c * w
        img_out[ws:we, hs:he, :] = np.array(img)

        img_out *= 255
        img_out = img_out.astype("int32")
        # img_out = np.clip(img_out, 0, 255)
        img = Image.fromarray(np.uint8(img_out))
        img.save("./gan_images/{}middle_out.png".format(num))


if __name__ == "__main__":
    batch_size = 8
    lr = [0.00001, 0.00001]
    bate_2 = [0.9, 0.9]
    ep = 100000  # epoch
    p = 0.8  # ada的p
    save_interval = [100, 1000]  # 多少轮保存图片, 保存模型
    data_path = r"E:\p站图片\database\face2head"
    reset = input("是否适用预设，batchsize：{0}，Epoch：{1}(y/n)".format(batch_size, ep))
    if reset in ("N", "n"):
        batch_size = input("batchsize: ")
        ep = input("Epochs：")
        batch_size = int(batch_size)
        ep = int(ep)

    isloadmodel = input("是否加载模型参数(y/n)")
    s_t = time.time()
    g = StyleGan(batch_size, lr, bate_2, data_path, p, istrain=False)
    if isloadmodel in ("Y", "y"):
        g.load_model(None)

    # 训练
    g.train(ep, save_interval=save_interval)

    # np.random.seed(int(time.time() % 100))
    # g.save_imgs("te")
    # g.G_mapping_avg()
    # g.change_W(0)
    # g.K_means_w()
    # g.predict_2w()

    # for i in range(1, 2):
    #     # g.middle_output(i)
    #     g.save_imgs("展示用", r=16, c=16)
    #     # g.predict_change(i)
    #     # g.middle_output(i)
    #     g.predict_change(i)
    #     g.predict_change(str(i) + 'A', move_mode='A')
    #     g.predict_change(str(i) + 'B', move_mode='B')
    #     print(i)

    # g.save_imgs(0)
    # g.predict_discriminator(4)

    e_t = time.time()
    print("epochs:{0},{1}s".format(ep, e_t - s_t))
    print((e_t - s_t) / ep, "s/epoch")
