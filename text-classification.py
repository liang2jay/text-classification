from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas
import xgboost
import numpy
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


class TextProcessing:
    def __init__(self):
        self.trainDF = pandas.DataFrame()

    def load(self):
        data = open('data/corpus').read()
        labels, texts = [], []
        for i, line in enumerate(data.split("\n")):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))

        # create a data frame using texts and label
        self.trainDF['text'] = texts
        self.trainDF['label'] = labels

        # split the data-set into training and validation data-sets
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(self.trainDF['text'], self.trainDF['label'], test_size=0.25)

        # label encode the target variable, Encode labels with value between 0 and n_classes-1.
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        return train_x, valid_x, train_y, valid_y

    def count_vector(self):
        train_x, valid_x, _, _ = self.load()
        # create a count vectorizer object, convert sentence to bag of word with frequency,
        # fit_transform: Learn the vocabulary dictionary and return term-document matrix.
        count_vector = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vector.fit(self.trainDF['text'])
        # transform the training and validation data using count vectorizer object
        x_train_count = count_vector.transform(train_x)
        x_valid_count = count_vector.transform(valid_x)

        return x_train_count, x_valid_count, count_vector

    def tf_idf_vector(self):
        train_x, valid_x, _, _ = self.load()
        tf_idf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tf_idf_vector.fit(self.trainDF['text'])
        x_train_tfidf = tf_idf_vector.transform(train_x)
        x_valid_tfidf = tf_idf_vector.transform(valid_x)

        return x_train_tfidf, x_valid_tfidf

    def word_embedding_vector(self):
        train_x, valid_x, _, _ = self.load()
        embeddings_index = {}
        # line = ['the real word',[word vector]]
        for i, line in enumerate(open('model\wiki-news-300d-1M.vec', encoding="ISO-8859-1")):
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:])

        # create a tokenizer and dictionary(word_index)
        token = text.Tokenizer()
        token.fit_on_texts(self.trainDF['text'])
        word_index = token.word_index

        # convert text to sequence of tokens by 'word_index'  and pad them to ensure equal length vectors
        train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
        valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

        # create token-embedding mapping, the word-number and word-vector
        embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_matrix[i] is not None:
                embedding_matrix[i] = embedding_vector
        return word_index, embedding_matrix, train_seq_x, valid_seq_x

    def topic_modeling_vector(self):
        x_train_count, _, count_vector = self.count_vector()
        # train a LDA Model
        lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
        X_topics = lda_model.fit_transform(x_train_count)
        topic_word = lda_model.components_
        vocab = count_vector.get_feature_names()

        # view the topic models
        n_top_words = 10
        topic_summaries = []
        for i, topic_dist in enumerate(topic_word):
            topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words + 1):-1]
            topic_summaries.append(' '.join(topic_words))

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        _, _, _, valid_y = self.load()
        # fit the training data-set on the classifier
        classifier.fit(feature_vector_train, label)
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        return metrics.accuracy_score(predictions, valid_y)

    def naive_bayes_classifier(self):
        # Naive Bayes on Count Vectors
        _, _, train_y, _ = self.load()
        x_train_count, x_valid_count, _ = self.count_vector()
        accuracy = self.train_model(naive_bayes.MultinomialNB(), x_train_count, train_y, x_valid_count)
        print("NB, Count Vectors: ", accuracy)

    def linear_classifier(self):
        # Linear Classifier on Count Vectors
        _, _, train_y, _ = self.load()
        x_train_count, x_valid_count, _ = self.count_vector()
        accuracy = self.train_model(linear_model.LogisticRegression(), x_train_count, train_y, x_valid_count)
        print("Linear Classifier on Count Vectors: ", accuracy)

        # Linear Classifier on Word Level TF IDF Vectors
        x_train_tfidf, x_valid_tfidf = self.tf_idf_vector()
        accuracy = self.train_model(linear_model.LogisticRegression(), x_train_tfidf, train_y, x_valid_tfidf)
        print("Linear Classifier on Word Level TF IDF Vectors: ", accuracy)

    def svm_classifier(self):
        # Support Vector Machine (SVM) is a supervised machine learning algorithm
        # SVM on TF IDF Vectors
        _, _, train_y, _ = self.load()
        x_train_tfidf, x_valid_tfidf = self.tf_idf_vector()
        accuracy = self.train_model(svm.SVC(), x_train_tfidf, train_y, x_valid_tfidf)
        print("SVM, tfidf Vectors: ", accuracy)

    def random_forest_classifier(self):
        # Random Forest models are a type of ensemble models, particularly bagging models
        # RF on Count Vectors
        _, _, train_y, _ = self.load()
        x_train_count, x_valid_count, _ = self.count_vector()
        x_train_tfidf, x_valid_tfidf = self.tf_idf_vector()
        accuracy = self.train_model(ensemble.RandomForestClassifier(), x_train_count, train_y, x_valid_count)
        print("Random Forest models Count Vectors, accuracy: ", accuracy)
        accuracy = self.train_model(ensemble.RandomForestClassifier(), x_train_tfidf, train_y, x_valid_tfidf)
        print("RF, WordLevel TF-IDF: ", accuracy)

    # Extreme Gradient Boosting on Count Vectors, TF IDF Vectors
    def boosting_classifier(self):
        _, _, train_y, _ = self.load()
        x_train_count, x_valid_count, _ = self.count_vector()
        x_train_tfidf, x_valid_tfidf = self.tf_idf_vector()
        accuracy = self.train_model(xgboost.XGBClassifier(), x_train_count.tocsc(), train_y, x_valid_count.tocsc())
        print("Extreme Gradient Boosting on Count Vectors: ", accuracy)
        accuracy = self.train_model(xgboost.XGBClassifier(), x_train_tfidf.tocsc(), train_y, x_valid_tfidf.tocsc())
        print("Extreme Gradient Boosting on TF-IDF: ", accuracy)

    def simple_neural_network(self):
        word_index, embedding_matrix, _, _ = self.word_embedding_vector()
        input_layer = layers.Input((70,))
        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        # Add the LSTM/GRU Layer
        lstm_layer = layers.LSTM(100)(embedding_layer)
        #          = layers.Bidirectional(layers.GRU(100))(embedding_layer)
        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

        return model

# Improving Text Classification Models
# Text Cleaning,
# In the feature engineering section, we generated a number of different feature vectors,combining them together
# Tuning the parameters
# Stacking different models and blending their outputs


if __name__ == '__main__':
    start = TextProcessing()
    start.simple_neural_network()



