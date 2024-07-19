from tensorflow.keras import regularizers

def l2_regularizer(l2=0.01):
    return regularizers.l2(l2)

