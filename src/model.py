"""
Figure 7 shows Flatten - 1164 - 100 - 50 - 10 - 1
Mine here goes Flatten - 100 - 50 - 10 - 1

Adding a 1164 layer costs 1152 * 1164 = 1.34M parameters
taking the model from 252k to 1.6M, a 6x increase all in one layer
Also it's already driving good and adding a 1.34M parameter layer to 2,816 training images invites overfitting

"""
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_nvidia_model(input_shape=(66, 200, 3), learning_rate=1e-4):
    inputs = Input(shape=input_shape)

    x = Conv2D(24, (5, 5), strides=(2, 2), activation='elu')(inputs)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)

    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

