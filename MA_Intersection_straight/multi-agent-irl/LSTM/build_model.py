# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:29:15 2020

@author: uqjsun9
"""

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout,Embedding, Conv1D,Bidirectional,Masking,Lambda
from keras.optimizers import RMSprop
# from keras.utils.data_utils import get_file
# from attention import attention_3d_block
import keras.layers
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
# from tcn import TCN

def lstm_model (features=18, Lr=0.01):
    print('Build lstm_model...')
    input_np = Input(shape=(None,  features), name='input_np')
    # input_tod = Input(shape=(None, 24), name='input_tod')
    # input_tt = Input(shape=(None, 1), name='input_tt')
    
    x = input_np
    x = Masking(mask_value = 0)(x)
    # x = Dropout(0.1)(x)
    # x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)
    # x = LSTM(256, return_sequences=True)(x)
    # x = LSTM(256, return_sequences=True)(x)
    
    x = Lambda((lambda xx: xx[:, 20:, :]))(x)
    
    # x = SeqSelfAttention(history_only=True)(x)
    # x = attention_3d_block(x)
    x = Dropout(0.2)(x)
    
    output_np = Dense(2, activation='tanh', name='output_np')(x)
    
    lstm_model = Model(input_np, output_np)
    
    
    optimizer = RMSprop(lr=Lr) #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # optimizer = keras.optimizers.Adam(lr=Lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    lstm_model.compile(loss='mse',   optimizer=optimizer)
    
    return lstm_model

def lstm_model_2out (features=169, Lr=0.01):
    print('Build model...')

    input_np = Input(shape=(None, features), name='input_np')
    input_tod = Input(shape=(None, 24), name='input_tod')
    
    x = keras.layers.concatenate([input_np, input_tod])
    
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    # x = Dropout(0.2)(x)
    x = Dropout(0.2)(x)
    output_np = Dense(features, activation='softmax', name='output_np')(x)
    output_tt = Dense(1, activation='relu', name='output_tt')(x)
    
    
    lstm_model_2out = Model([input_np, input_tod], [output_np, output_tt])
    
    losses= {'output_np': 'categorical_crossentropy', 'output_tt': 'mse'}
    loss_weights={'output_np': 1., 'output_tt': 100.}
    
    optimizer = RMSprop(learning_rate=Lr) #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    lstm_model_2out.compile(loss=losses, loss_weights=loss_weights,  optimizer=optimizer)

    return lstm_model_2out

def lstm_attention_model (features=169, Lr=0.01):
    print('Build lstm_attention_model...')

    input_np = Input(shape=(None, features), name='input_np')
    input_tod = Input(shape=(None, 24), name='input_tod')
    input_tt = Input(shape=(None, 1), name='input_tt')
    input_sp = Input(shape=(None, features), name='input_np')
    input_dow = Input(shape=(None, 7), name='input_dow')
    # input_tt = SeqSelfAttention()(input_tt)
    
    x = keras.layers.concatenate([input_np,input_tod,input_dow, input_tt])
    x = Embedding(mask_zero=True)(x)      
    x = LSTM(128, return_sequences=True)(x)
    x = SeqSelfAttention(history_only=True)(x)    
    x = Dropout(0.2)(x)
    
 
    # x2 = SeqSelfAttention(attention_width=1)(x2)
    
    # x2 = Dropout(0.2)(x2)
    
    output_np = Dense(features, activation='softmax', name='output_np')(x)
    
    x2 = keras.layers.concatenate([x, input_tt])
    x2 = Embedding(mask_zero=True)(x2)
    x2 = LSTM(128, return_sequences=True)(x2)
    x2 = SeqSelfAttention(history_only=True)(x2)    
    x2 = Dropout(0.2)(x2)
    output_tt = Dense(1, activation='relu', name='output_tt')(x)
    
    
    lstm_attention_model = Model([input_np, input_tod,input_tt,input_dow], [output_np, output_tt])
    
    losses= {'output_np': 'categorical_crossentropy', 'output_tt': 'mae'}
    loss_weights={'output_np': 1., 'output_tt': 8}
    
    optimizer = RMSprop(learning_rate=Lr) 
    # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    lstm_attention_model.compile(loss=losses, loss_weights=loss_weights,  optimizer=optimizer)

    return lstm_attention_model

def bilstm_model (features=169, Lr=0.01):
    print('Build bilstm_model...')

    input_np = Input(shape=(None, features), name='input_np')
    input_tod = Input(shape=(None, 24), name='input_tod')
    input_tt = Input(shape=(None, 1), name='input_tt')
    input_dow = Input(shape=(None, features), name='input_dow')
    # input_tt = SeqSelfAttention()(input_tt)
    
    x = keras.layers.concatenate([input_np,input_tod,input_dow,input_tt])
          
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = SeqSelfAttention(attention_width=1)(x)    
    x = Dropout(0.2)(x)
    
 
    # x2 = SeqSelfAttention(attention_width=1)(x2)
    
    # x2 = Dropout(0.2)(x2)
    
    output_np = Dense(features, activation='softmax', name='output_np')(x)
    
    x2 = keras.layers.concatenate([x, input_tt])
    x2 = LSTM(128, return_sequences=True)(x2)
    x2 = SeqSelfAttention(attention_width=1)(x2)    
    x2 = Dropout(0.2)(x2)
    output_tt = Dense(1, activation='relu', name='output_tt')(x)
    
    
    bilstm_model = Model([input_np, input_tod,input_tt], [output_np, output_tt])
    
    losses= {'output_np': 'categorical_crossentropy', 'output_tt': 'mae'}
    loss_weights={'output_np': 1., 'output_tt': 8}
    
    optimizer = RMSprop(learning_rate=Lr) 
    # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    bilstm_model.compile(loss=losses, loss_weights=loss_weights,  optimizer=optimizer)

    return bilstm_model

# def TCN_model (features=169, Lr=0.01):
#     print('Build TCN_model...')

#     input_np = tf.keras.Input(shape=(None, features), name='input_np')
#     input_tod = Input(shape=(None, 24), name='input_tod')
#     input_tt = Input(shape=(None, 1), name='input_tt')
#     # input_tt = SeqSelfAttention()(input_tt)
    
#     # x = keras.layers.concatenate([input_np, input_tod])
          
#     x = TCN(return_sequences=True)(input_np)
    
     
    
#     output_np = tf.keras.layers.Dense(features, activation='softmax', name='output_np')(x)
#     # output_tt = Dense(1, activation='relu', name='output_tt')(x2)
    
    
#     TCN_model = tf.keras.Model(input_np, output_np)
    
#     losses= {'output_np': 'categorical_crossentropy', 'output_tt': 'mae'}
#     loss_weights={'output_np': 1., 'output_tt': 1.}
    
#     optimizer = RMSprop(learning_rate=Lr) #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
#     TCN_model.compile(loss= 'categorical_crossentropy',  optimizer='adam')

#     return TCN_model