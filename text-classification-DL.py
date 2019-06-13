from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Bidirectional, CuDNNLSTM, CuDNNGRU, GlobalAveragePooling1D
from keras import backend as K
from keras.layers import LSTM, Lambda
import numpy as np
import os


class TextCNN:
    def __init__(self, max_length, max_features, embedding_dims, dense_num, last_activation):
        self.max_len = max_length
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.dense_num = dense_num
        self.last_activation = last_activation

    def get_model(self):
        input_layer = Input((self.max_len,))
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)(input_layer)
        convolution_layer = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convolution_layer.append(c)
        concatenate = Concatenate()(convolution_layer)

        output = Dense(self.dense_num, activation=self.last_activation)(concatenate)
        model = Model(inputs=input_layer, outputs=output)
        return model


# only install tensor_flow-GPU in your PC is acceptable when component need it. GPU is much faster
class TextBiRNN:
    def __init__(self, max_length, max_features, embedding_dims, dense_num, last_activation):
        self.max_len = max_length
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.dense_num = dense_num
        self.last_activation = last_activation

    def get_model(self):
        input_layer = Input((self.max_len,))
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)(input_layer)
        BiLSTM = Bidirectional(CuDNNLSTM(128))(embedding)
        output = Dense(self.dense_num, activation=self.last_activation)(BiLSTM)
        model = Model(inputs=input_layer, outputs=output)
        return model


class RecurrentCNN:
    def __init__(self, max_length, max_features, embedding_dims, dense_num, last_activation):
        self.max_len = max_length
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.dense_num = dense_num
        self.last_activation = last_activation

    def get_model(self):
        input_current = Input((self.max_len,))
        input_left = Input((self.max_len,))
        input_right = Input((self.max_len,))

        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)

        embedding_current = embedding(input_current)
        embedding_left = embedding(input_left)
        embedding_right = embedding(input_right)

        x_left = LSTM(128, return_sequences=True)(embedding_left)
        x_right = LSTM(128, return_sequences=True, go_backwards=True)(embedding_right)
        x_right = Lambda(lambda x_: K.reverse(x_, axes=1))(x_right)
        x_r = Concatenate(axis=2)([x_left, embedding_current, x_right])

        x = Conv1D(64, kernel_size=1, activation='tanh')(x_r)
        x = GlobalMaxPooling1D()(x)

        output = Dense(self.dense_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_current, input_left, input_right], outputs=output)
        return model


class RecurrentCNNPro:
    def __init__(self, max_length, max_features, embedding_dims, dense_num, last_activation):
        self.max_len = max_length
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.dense_num = dense_num
        self.last_activation = last_activation

    def get_model(self):
        input_layer = Input((self.max_len,))
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.max_len)(input_layer)
        x_context = Bidirectional(CuDNNLSTM(128, return_sequences=True))(embedding)
        x = Concatenate()([embedding, x_context])

        convs = []
        for kernel_size in range(1, 5):
            conv = Conv1D(128, kernel_size, activation='relu')(x)
            convs.append(conv)
        poolings = [GlobalAveragePooling1D()(conv) for conv in convs] + [GlobalMaxPooling1D()(conv) for conv in convs]
        x = Concatenate()(poolings)

        output = Dense(self.dense_num, activation=self.last_activation)(x)
        model = Model(inputs=input_layer, outputs=output)
        return model


class Classification:
    def __init__(self):
        self.max_features = 5000
        self.max_len = 200
        self.batch_size = 32
        self.embedding_dims = 100
        self.epochs = 10
        self.dense = 1
        self.activation = 'sigmoid'

    def load_data(self):
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Pad sequences (samples x time)...')
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('Build model...')
        return x_train, y_train, x_test, y_test

    def load_data_concat_network(self):
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Pad sequences (samples x time)...')
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
        x_test = sequence.pad_sequences(x_test, self.max_len)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        print('Prepare input for model...')
        x_train_current = x_train
        x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]])
        x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)])

        x_test_current = x_test
        x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
        x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])
        print('x_train_current shape:', x_train_current.shape)
        print('x_train_left shape:', x_train_left.shape)
        print('x_train_right shape:', x_train_right.shape)
        print('x_test_current shape:', x_test_current.shape)
        print('x_test_left shape:', x_test_left.shape)
        print('x_test_right shape:', x_test_right.shape)

        return [x_train_current, x_train_left, x_train_right], y_train, [x_test_current, x_test_left, x_test_right], y_test

    def training_start(self, neural_network, load_data_method):
        x_train, y_train, x_test, y_test = load_data_method
        model = neural_network(self.max_len, self.max_features, self.embedding_dims, self.dense, self.activation).get_model()
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
        checkpoint = ModelCheckpoint(filepath=os.getcwd(), monitor='val_loss', period=5)
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  callbacks=[early_stopping, checkpoint],
                  validation_data=(x_test, y_test))
        print('Test...')
        result = model.predict(x_test)
        print(result)


if __name__ == '__main__':
    start = Classification()
    # start.training_start(TextCNN, start.load_data())
    # start.training_start(TextBiRNN, start.load_data())
    # start.training_start(RecurrentCNN, start.load_data_concat_network())
    # start.training_start(RecurrentCNNPro, start.load_data())



