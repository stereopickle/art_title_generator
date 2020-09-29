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
import json

import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

    
def get_keys(dict_):
    ''' 
    Helper to return a list of keys 
    given dictionary
    '''
    return list(dict_.keys())

def get_vals(dict_):
    ''' 
    Helper to return a list of values 
    given dictionary
    '''
    return list(dict_.values())
    
def get_features(features_dict, img_ind):
    ''' 
    Helper to return feature values 
    given feature dictionary and index
    '''
    if isinstance(img_ind, list):
        return [features_dict[x][0] for x in img_ind]
    elif isinstance(img_ind, str):
        return features_dict[img_ind][0]
    else:
        print('img_ind must be a list or string type')
        return None
    
def get_text(dictionary, img_ind):
    ''' 
    Helper to return a list of description 
    given an index 
    '''
    return dictionary[img_ind]    

class sequence_generator:
    ''' 
    Returns a sequence_generator object
    '''
    def __init__(self, dictionary, features):
        ''' INPUT: a dictionary of descriptions and features '''
        self.dictionary = dictionary
        self.features = features
        self.img_index = get_keys(self.dictionary)
        self.texts = get_vals(self.dictionary)

    def update_selection(self, list_):
        ''' 
        INPUT: select list of image indices
        Create selector, and subsets (select_dictionary, select_img_inds, select_texts)
        '''
        self.selector = list_
        self.select_dictionary = {k: v for k, v in self.dictionary.items() if (k in list_) & (k in self.features)}
        self.select_img_inds = get_keys(self.select_dictionary)
        self.select_texts = get_vals(self.select_dictionary)
    
    def sequence_process(self, dict_):
        ''' Helper to process breakdown on all select dictionary '''
        X1, X2, Y = [], [], []

        def breakdown_sequence(list_):
            ''' Helper to return a list of breakdown sequences and the output '''
            x, y = [], []
            for i in range(1, len(list_)):
                x.append(list_[:i])
                y.append(list_[i])
            return x, y
        
        for ind, texts in dict_.items():
            sequences = self.tokenizer.texts_to_sequences(texts)
            
            for seq in sequences:
                x, y = breakdown_sequence(seq)

                X1.extend(np.repeat(ind, len(y)))
                X2.extend(x)
                Y.extend(y)

        return X1, X2, Y

    def train_generator(self, train_list):
        '''
        INPUT a list of training ids, 
        RETURN image inputs, text inputs, and outputs
        ASSIGN max_length and vocab size
        '''
        self.update_selection(train_list)

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(np.concatenate(self.select_texts))
        self.num_vocab = len(self.tokenizer.word_index)+1
        
        dict_ = self.select_dictionary
        

        X1, X2, Y = self.sequence_process(dict_)
        
        X2 = pad_sequences(X2)
        self.max_length = X2.shape[1]
    
        Y = to_categorical(Y, self.num_vocab)
        X1 = get_features(self.features, X1)
        
        return np.array(X1), np.array(X2), np.array(Y)

    def validation_generator(self, val_list):
        '''
        INPUT a list of validation ids, 
        RETURN image inputs, text inputs and outputs
        '''
        self.update_selection(val_list)
        
        dict_ = self.select_dictionary

        X1, X2, Y = self.sequence_process(dict_)
        X2 = pad_sequences(X2, maxlen = self.max_length)
        Y = to_categorical(Y, num_classes = self.num_vocab)
        X1 = get_features(self.features, X1)

        return np.array(X1), np.array(X2), np.array(Y)
    
    def get_num_vocab(self):
        return self.num_vocab
    def get_max_length(self):
        return self.max_length
    def get_tokenizer(self):
        return self.tokenizer


def _parse_args():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu_count', type=int, default=os.environ['SM_NUM_GPUS'])
    #parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    return parser.parse_known_args()

if __name__ == '__main__':
     
    
    args, _ = _parse_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    pkl_dir    = args.train

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

    train_X1, train_X2, train_Y = processor.train_generator(train_list_full[0:100])
    val_X1, val_X2, val_Y = processor.validation_generator(val_list_full[0:20])
    
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
    
    
    
    # save Keras model for Tensorflow Serving
    if args.current_host == args.hosts[0]:
        model.save(os.path.join(args.sm_model_dir, '001'))