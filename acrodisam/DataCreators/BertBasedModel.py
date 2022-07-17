'''
Created on Feb 8, 2020

@author: jpereira
'''

import os


import bert
import tensorflow as tf
import sentencepiece as spm
from string_constants import FOLDER_ALBERT


def getAlbertModel(preTrainedModel = "albert_base"):
    model_params = bert.albert_params(preTrainedModel)
    l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
    return l_bert
       
def loadAlbertWeights(l_bert, preTrainedModel = "albert_base"):
    model_dir    = bert.fetch_tfhub_albert_model(preTrainedModel, FOLDER_ALBERT)
    bert.load_albert_weights(l_bert, model_dir)      # should be called after model.build()
        
def getAlbertSentenceProcessor(preTrainedModel = "albert_base"):
    model_dir    = bert.fetch_tfhub_albert_model(preTrainedModel, FOLDER_ALBERT)

    #sp_model = tf.io.gfile.glob(os.path.join(FOLDER_ALBERT + preTrainedModel, "assets/*"))[0]
    sp_model = tf.io.gfile.glob(os.path.join(model_dir, "assets/*"))[0]
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model)
    return sp
