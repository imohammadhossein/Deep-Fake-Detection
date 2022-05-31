from keras.layers.core import Dense
from keras.layers import Input, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import SGD, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import GRU
from keras import utils
import numpy as np
import time
import argparse
from keras.engine import InputSpec
from keras.engine.topology import Layer


class TemporalMaxPooling(Layer):

    def __init__(self, **kwargs):
        super(TemporalMaxPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if mask is None:
            mask = K.sum(K.ones_like(x), axis=-1)
        if K._BACKEND == "tensorflow":
            mask = K.expand_dims(mask, axis=-1)
            mask = K.tile(mask, (1, 1, K.int_shape(x)[2]))
            masked_data = K.tf.where(
                K.equal(mask, K.zeros_like(mask)), K.ones_like(x) * -np.inf, x
            )  # if masked assume value is -inf
            return K.max(masked_data, axis=1)
        else:  # theano backend
            mask = mask.dimshuffle(0, 1, "x")
            masked_data = K.switch(K.eq(mask, 0), -np.inf, x)
            return masked_data.max(axis=1)

    def compute_mask(self, input, mask):
        # do not pass the mask to the next layers
        return None


def lstm_model(train_data):
    # Model definition
    main_input = Input(
        shape=(train_data.shape[1],
               train_data.shape[2]),
        name="main_input"
    )
    headModel = Bidirectional(LSTM(256, return_sequences=True))(main_input)
    headModel = LSTM(32)(main_input)
    headModel = TemporalMaxPooling()(headModel)
    headModel = TimeDistributed(Dense(512))(headModel)
    headModel = Bidirectional(LSTM(512, dropout=0.2))(main_input)
    headModel = LSTM(256)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform"
    )(headModel)
    model = Model(inputs=main_input, outputs=predictions)

    # Model compilation
    # opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / EPOCHS)
    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model



start = time.time()

ap = argparse.ArgumentParser()
ap.add_argument(
    "-e", "--epochs", required=True, type=int,
    help="Number of epochs", default=25
)
ap.add_argument(
    "-w",
    "--weights_save_name",
    required=True,
    type=str,
    help="Model weights name"
)
ap.add_argument(
    "-b", "--batch_size", required=True, type=int,
    help="Batch size", default=32
)
args = ap.parse_args()

# Training dataset loading
train_data = 
train_label = 
train_label = 

# Train validation split
trainX, valX, trainY, valY = train_test_split(
    train_data, train_label, shuffle=True, test_size=0.1
)

model = lstm_model(train_data)

trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.trainable_weights)])
)
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
)

# Number of trainable and non-trainable parameters
print("Total params: {:,}".format(trainable_count + non_trainable_count))
print("Trainable params: {:,}".format(trainable_count))
print("Non-trainable params: {:,}".format(non_trainable_count))

# Keras backend
model_checkpoint = ModelCheckpoint(
    "trained_wts/CNN_LSTM.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0)

print("Training is going to start in 3... 2... 1... ")

# Model training
H = model.fit(
    trainX,
    trainY,
    validation_data=(valX, valY),
    batch_size=args.batch_size,
    epochs=args.epochs,
    shuffle=True,
    callbacks=[model_checkpoint, stopping],
)