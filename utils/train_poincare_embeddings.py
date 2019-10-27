# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:42:20 2019

@author: dreww
"""

# simple wrapper for gensim's poincare model
# source: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/poincare.py

# import libraries
from gensim.models.poincare import PoincareModel, PoincareRelations
import logging
logging.basicConfig(level=logging.INFO)

def train_embeddings(input_path, # path to input edge relations
                     delimiter, # input file delim
                     output_path, # path to output embedding vectors 
                     size=2, # embed dimension
                     alpha=0.1, # learning rate
                     burn_in=10, # burn in train rounds
                     burn_in_alpha=0.01, # burn in learning rate
                     workers=1, # number of training threads used
                     negative=10, # negative sample size
                     epochs=100, # training rounds
                     print_every=500, # print train info
                     batch_size=10): # num samples in batch
    
    # load file with edge relations between entities
    relations = PoincareRelations(file_path=input_path, delimiter=delimiter)
    
    # train model
    model = PoincareModel(train_data=relations, size=size, alpha=alpha, burn_in=burn_in,
                          burn_in_alpha=burn_in_alpha, workers=workers, negative=negative)
    model.train(epochs=epochs, print_every=print_every,batch_size=batch_size)
    
    # save output vectors
    model.kv.save_word2vec_format(output_path)
    
    return


    
    
    