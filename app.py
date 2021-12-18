# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:52:10 2021

@author: riche
"""

import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import gradio as gr 

X_train = train.copy()
y_train = X_train.pop('label')


scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)

model = load_model('digit_recognizer_modeldef.h5')

def sketch_recognition(img):
  # Implement sketch recognition model here...
  # Return labels and confidences as dictionary
    img = img.reshape((1, 784))
    img = scale.transform(img.reshape(1, -1))
    preds = model.predict(np.array(img).reshape((1, 28, 28, 1))).tolist()[0]
    return {str(i): preds[i] for i in range(10)}

interface = gr.Interface(fn=sketch_recognition, inputs="sketchpad", outputs=gr.outputs.Label()).launch(share=True)