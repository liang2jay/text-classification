from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


class TextProcessing:
    def __init__(self):
        data = open('data/corpus').read()
        labels, texts = [], []
        for i, line in enumerate(data.split("\n")):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))

        # create a dataframe using texts and lables
        trainDF = pandas.DataFrame()
        trainDF['text'] = texts
        trainDF['label'] = labels

        # split the dataset into training and validation datasets
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        self.train_y = encoder.fit_transform(train_y)
        self.valid_y = encoder.fit_transform(valid_y)

        # load the pre-trained word-embedding vectors
        embeddings_index = {}
        for i, line in enumerate(open('model\wiki-news-300d-1M.vec', encoding="ISO-8859-1")):
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:])

        # create a tokenizer
        token = text.Tokenizer()
        token.fit_on_texts(trainDF['text'])
        self.word_index = token.word_index

        # convert text to sequence of tokens and pad them to ensure equal length vectors
        self.train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
        self.valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

        # create token-embedding mapping
        self.embedding_matrix = numpy.zeros((len(self.word_index) + 1, 300))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def create_cnn(self):
        # Add an Input Layer
        input_layer = layers.Input((70,))
        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)
        # Add the pooling layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

    def create_lstm(self):
        # add an input layer
        input_layer = layers.Input((70,))
        # add the word embedding layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        # Add the LSTM Layer or gru
        # gru_layer = layers.GRU(100)(embedding_layer)
        # lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)
        lstm_layer = layers.LSTM(100, return_sequences=True)(embedding_layer)
        lstm_layer_2 = layers.LSTM(100)(lstm_layer)
        # Add the output layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer_2)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
        # compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        return model

    def create_rcnn(self):
        # add an input layer
        input_layer = layers.Input((70,))

        # add the word embedding layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # add the recurrent layer
        rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)

        # add the convolutional layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(rnn_layer)

        # add the pooling layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        predictions = predictions.argmax(axis=-1)
        return metrics.accuracy_score(predictions, self.valid_y)


if __name__ == '__main__':
    start = TextProcessing()
    accuracy = start.train_model(start.create_rcnn(), start.train_seq_x, start.train_y, start.valid_seq_x)
    print(accuracy)










