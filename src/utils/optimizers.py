from tensorflow.keras.optimizers import Adam, SGD

def get_optimizer(name='adam', learning_rate=0.001):
    if name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif name == 'sgd':
        return SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type")

