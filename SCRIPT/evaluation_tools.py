# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:31:45 2020

@author: Eunjoo
This script contains all code related to evaluation
including plotting results and making predictions

"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.style.use('fivethirtyeight')


def plot_performance(hist):
    ''' 
    function to compare training and validation 
    Input: model with epochs info
    '''
    hist_ = hist.history
    epochs = hist.epoch
    
    plt.figure()
    plt.plot(epochs, hist_['loss'], label='Training loss')
    plt.plot(epochs, hist_['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    
def ind2word(ind, tokenizer):
    '''
    function to convert word ids to word
    Input: int 
    Output: str
    '''
    return tokenizer.index_word[ind]



def caption_generator(img_ind, features, tokenizer, max_length, model):
    ''' 
    helper function to return the prediction 

    Input: image id (str), feature_dict, tokenizer, max_length(int)
    Output: a caption
    '''
    
    img_feats = get_features(feature_dict, img_ind)
    img_feats = np.expand_dims(img_feats, axis = 0)
    current_int = tokenizer.texts_to_sequences(['seqini'])
    fin_int = tokenizer.texts_to_sequences(['seqfin'])[0]
    
    # iterate each sequence and predict the next word
    for i in range(max_length):
        current_seq = pad_sequences(current_int, maxlen = max_length)
        next_int = np.argmax(model.predict([img_feats, current_seq]))
        if next_int != fin_int:
            current_int = [current_int[0] + [next_int]]
        else: break
    
    # now translate it into the word
    return ' '.join([ind2word(x, tokenizer) for x in current_seq[0] if x != 0][1:])

class descriptor:
    '''
    Input features, tokenizer, processor, model, img_dir
    Return a descriptor object
    '''
    def __init__(self, features, tokenizer, processor, model, img_dir):
        self.features = features
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.max_length = processor.get_max_length()
        self.num_vocab = processor.get_num_vocab()
        self.tokenizer = processor.get_tokenizer()
        self.img_dir = img_dir
        
    def test_one_image(self, img_id):
        ''' print generated caption and image given an id'''
        print(caption_generator(img_id, self.features, self.tokenizer, self.max_length))
        img = mpimg.imread(f'{self.img_dir}/{img_id}.jpg')
        plt.imshow(img)
        plt.grid(False)
        plt.show()
    
    def update_model(self, newmodel):
        self.model = newmodel
    
    def update_directory(self, new_dir):
        self.img_dir = new_dir
        
    def test_random_image(self, img_inds):
        ''' print random image and generated caption '''
        rand_id = np.random.choice(img_inds, 1)[0]
        self.test_one_image(rand_id)
        
def get_bleu(img_inds, feature_dict, tokenizer, max_length, model, text_ref_dict):
    ''' 
    Input takes image index, feature dictionary, 
    tokenizer, max length of tokens, model, 
    and description dictionary 
    Output predictions and BLEU score for 1-gram to 4-gram 
    '''
    prediction_list = {}
    n = len(img_inds)
    hypotheses = []
    references = []
    for i, ind in enumerate(img_inds):
        caption = caption_generator(ind, 
                                    feature_dict, 
                                    tokenizer, 
                                    max_length, 
                                    model)
        prediction_list[ind] = caption
        hypotheses.append(caption.split())
        
        ref = get_text(text_ref_dict, ind)
        references.append([x.split()[1:-1] for x in ref])
        
        print(i+1, '/', n, 'complete')
    bleu_1 = corpus_bleu(references, hypotheses, weights = (1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights = (.5, .5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights = (.3, .3, .3, 0))
    bleu_4 = corpus_bleu(references, hypotheses)

    print(f'1-gram BLEU: {round(bleu_1, 4)}')
    print(f'2-gram BLEU: {round(bleu_2, 4)}')
    print(f'3-gram BLEU: {round(bleu_3, 4)}')
    print(f'4-gram BLEU: {round(bleu_4, 4)}')
    return prediction_list, (bleu_1, bleu_2, bleu_3, bleu_4)
