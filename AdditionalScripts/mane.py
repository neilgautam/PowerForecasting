import keras.backend as K

def mane_loss(y_true, y_pred):
    return K.sum((y_true-y_pred)/y_true)
