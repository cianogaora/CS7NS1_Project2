#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tflite_runtime.interpreter as tflite 
#import tensorflow.keras as keras
from PIL import Image


def decode(characters, y):
#    print(y)
    y = numpy.argmax(numpy.array(y), axis=1)
    return ''.join([characters[x] for x in y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    img_files = os.listdir(args.captcha_dir)
    img_files = sorted(img_files)

#    with tflite.device('/cpu:0'):
    with open(args.output, 'w',newline='\n') as output_file:
        json_file = open(args.model_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # model = keras.models.model_from_json(loaded_model_json)
        # model.load_weights(args.model_name+'.h5')
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
        #               metrics=['accuracy'])
        # keras_model = tf.keras.models.load_model(args.model_name + '.h5')
        # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # tflite_model = converter.convert()
        interpreter = tflite.Interpreter(args.model_name+'.tflite')
        interpreter.allocate_tensors()
        # tflite_model.allocate_tensors()

        for x in img_files:
            # load image and preprocess it
            raw_data = Image.open(os.path.join(args.captcha_dir, x))

            # rgb_data = Image.fromarray(raw_data)

            image = numpy.array(raw_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            image = image.astype('float32')
            # print(image_data)
            interpreter.set_tensor(input_details[0]['index'],image)
            interpreter.invoke()
            captcha = ""
            captcha_final = ""
            # print(output_details)
            output_data_0 = interpreter.get_tensor(output_details[3]['index'])
            output_data_1 = interpreter.get_tensor(output_details[5]['index'])
            output_data_2 = interpreter.get_tensor(output_details[0]['index'])
            output_data_3 = interpreter.get_tensor(output_details[4]['index'])
            output_data_4 = interpreter.get_tensor(output_details[2]['index'])
            output_data_5 = interpreter.get_tensor(output_details[1]['index'])

            final1 = decode(captcha_symbols, output_data_0)
            final2 = decode(captcha_symbols, output_data_1)
            final3 = decode(captcha_symbols, output_data_2)
            final4 = decode(captcha_symbols, output_data_3)
            final5 = decode(captcha_symbols, output_data_4)
            final6 = decode(captcha_symbols, output_data_5)
            captcha = captcha + final1 + final2 + final3 + final4 + final5 +final6
            # for i in range(4):
            #     # captcha = ""
            #     output_data = interpreter.get_tensor(output_details[i]['index'])
            #     final = decode(captcha_symbols,output_data)
            #     captcha = captcha + final
            #     print(final)
            # # captcha = ""
            # #     for i in  (final):
            #     if (final != '/'):
            #         captcha = captcha + final
            # print(captcha)
            for i in captcha:
                if(i != '/'):
                    captcha_final = captcha_final + i

            output_file.write(x + "," + captcha_final + "\n")

            print('Classified ' + x)

if __name__ == '__main__':
    main()
