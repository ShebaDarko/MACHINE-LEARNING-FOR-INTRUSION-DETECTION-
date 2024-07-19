from import_libraries import *

def create_mlp_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(60, activation='relu', input_shape=input_shape))
    model.add(Dense(num_classes, activation='softmax'))
    return model

