# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:40:41 2020

@author: Eunjoo

This script will run the full model steps
Base structure is following AWS blog post here: 
https://aws.amazon.com/blogs/machine-learning/train-and-deploy-keras-models-with-tensorflow-and-apache-mxnet-on-amazon-sagemaker/

"""

import argparse, os
import numpy as np
import pickle

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, LSTM, add
from keras.optimizers import Adam
from keras.utils import multi_gpu_model



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default='/tmp')
    parser.add_argument('--photos-dir', type=str, default='data')
    parser.add_argument('--img-dir', type=str, default='data')
    parser.add_argument('--pkl-dir', type=str, default='data')

    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    img_dir    = args.img_dir
    pkl_dir    = args.pkl_dir
    
    desc_path        = os.path.join(pkl_dir, 'full_descriptions.pkl')
    feat_path        = os.path.join(pkl_dir, 'full_features.pkl')
    train_list_path  = os.path.join(pkl_dir, 'train_list_full.pkl')
    val_list_path    = os.path.join(pkl_dir, 'val_list_full.pkl')
    test_list_path   = os.path.join(pkl_dir, 'test_list.pkl')
    test_art_path    = os.path.join(pkl_dir, 'test_list_art.pkl')

    with open(train_list_path, 'rb') as fp:
        train_list_full = pickle.load(fp)
    
    with open(val_list_path, 'rb') as fp:
        val_list_full = pickle.load(fp)
        
    with open(test_list_path, 'rb') as fp:
        test_list = pickle.load(fp)
        
    with open(test_art_path, 'rb') as fp:
        test_list_art = pickle.load(fp)

    with open(desc_path, 'rb') as fp:
        descriptions = pickle.load(fp)
    
    with open(feat_path, 'rb') as fp:
        features = pickle.load(fp)
        
        
    # generate inputs and outputs
    processor = sequence_generator(descriptions, features)
    train_X1, train_X2, train_Y = processor.train_generator(train_list_full)
    val_X1, val_X2, val_Y = processor.validation_generator(val_list_full)
    
    # get params
    tokenizer = processor.get_tokenizer()
    max_length = processor.get_max_length()
    num_vocab = processor.get_num_vocab()
    
    # model architecture
    #first path
    in1 = Input(shape = (4032,))
    img_layer1 = Dropout(0.5)(in1)
    img_layer2 = Dense(512, activation = 'relu')(img_layer1)
    
    # second path
    in2 = Input(shape=(max_length,))
    text_layer1 = Embedding(num_vocab, 512, mask_zero = True)(in2)
    text_layer2 = Dropout(0.5)(text_layer1)
    text_layer3 = LSTM(512, dropout = 0.5, return_sequences = True)(text_layer2)
    text_layer4 = LSTM(512, dropout = 0.5)(text_layer3)
    
    # outputting
    output_layer1 = add([img_layer2, text_layer4])
    output_layer2 = Dense(512, activation = 'relu')(output_layer1)
    output = Dense(num_vocab, activation = 'softmax')(output_layer2)
    
    # compile model
    model = Model(inputs = [in1, in2], outputs = output)

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    model.compile(loss = 'categorical_crossentropy', 
             optimizer = Adam(lr=lr))   
    
    history = model.fit([train_X1, train_X2], train_Y, 
              batch_size = batch_size,
              epochs=epochs, 
              validation_data = ([val_X1, val_X2], val_Y),
              verbose = 1
              )
    
    plot_performance(history)
    get_bleu(img_inds, feature_dict, tokenizer, max_length, model, text_ref_dict):

    
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])
    
    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
