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
import tensorflow as tf
# import tensorflow.keras as keras

def decode(characters, y):
    y = numpy.argmax(y)
    return characters[y]

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

    with tf.device('/cpu:0'):
        with open(args.output, 'w', newline="\n") as output_file:
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            converter = tf.lite.TFLiteConverter.from_keras_model_file(args.model_name+'.h5')
            model = converter.convert()
            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_content=model)
            interpreter.allocate_tensors()

            for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, 64, 128, w])
       
                # if image.shape != (64, 128, 3):
                
                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Test the model on input data.
                input_shape = input_details[0]['shape']

                # Use same image as Keras model
                input_data = numpy.array(image, dtype=numpy.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                print(output_details)
                captcha = ""
                for i in range(4):
                    prediction = interpreter.get_tensor(output_details[i]['index'])
                    cap = decode(captcha_symbols, prediction)
                    if(cap != '/'):
                        captcha = captcha + cap
                output_file.write(x + "," + captcha + "\n")

                print('Classified ' + x)

if __name__ == '__main__':
    main()
