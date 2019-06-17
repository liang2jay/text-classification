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

        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(trainDF['text'])
        # transform the training and validation data using count vectorizer object
        self.xtrain_count = count_vect.transform(train_x)
        self.xvalid_count = count_vect.transform(valid_x)

        # word level tf-idf
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(trainDF['text'])
        self.xtrain_tfidf = tfidf_vect.transform(train_x)
        self.xvalid_tfidf = tfidf_vect.transform(valid_x)

        # load the pre-trained word-embedding vectors
        embeddings_index = {}
        for i, line in enumerate(open('model\wiki-news-300d-1M.vec', encoding="ISO-8859-1")):
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:])

        # create a tokenizer
        token = text.Tokenizer()
        token.fit_on_texts(trainDF['text'])
        word_index = token.word_index

        # convert text to sequence of tokens and pad them to ensure equal length vectors
        self.train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
        self.valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

        # create token-embedding mapping
        embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    def train_model(self, classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)

        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)

        if is_neural_net:
            predictions = predictions.argmax(axis=-1)

        return metrics.accuracy_score(predictions, self.valid_y)


if __name__ == '__main__':
    start = TextProcessing()
    result = start.train_model(naive_bayes.MultinomialNB(), start.xtrain_count, start.train_y, start.xvalid_count)
    print("naive_bayes, Count Vectors: ", result)
    result = start.train_model(naive_bayes.MultinomialNB(), start.xtrain_tfidf, start.train_y, start.xvalid_tfidf)
    print("naive_bayes, WordLevel TF-IDF: ", result)
    print('-------------------------------------------------------------------------------------------------------')
    result = start.train_model(linear_model.LogisticRegression(), start.xtrain_count, start.train_y, start.xvalid_count)
    print("LogisticRegression, Count Vectors: ", result)
    result = start.train_model(linear_model.LogisticRegression(), start.xtrain_tfidf, start.train_y, start.xvalid_tfidf)
    print("LogisticRegression, WordLevel TF-IDF: ", result)
    print('-------------------------------------------------------------------------------------------------------')
    result = start.train_model(svm.SVC(), start.xtrain_count, start.train_y, start.xvalid_count)
    print("Support Vector Machine, Count Vectors: ", result)
    result = start.train_model(svm.SVC(), start.xtrain_tfidf, start.train_y, start.xvalid_tfidf)
    print("Support Vector Machine, WordLevel TF-IDF: ", result)
    print('-------------------------------------------------------------------------------------------------------')
    result = start.train_model(ensemble.RandomForestClassifier(), start.xtrain_count, start.train_y, start.xvalid_count)
    print("RandomForest, Count Vectors: ", result)
    result = start.train_model(ensemble.RandomForestClassifier(), start.xtrain_tfidf, start.train_y, start.xvalid_tfidf)
    print("RandomForest, WordLevel TF-IDF: ", result)
    print('-------------------------------------------------------------------------------------------------------')
    result = start.train_model(xgboost.XGBClassifier(), start.xtrain_count, start.train_y, start.xvalid_count)
    print("xgboost, Count Vectors: ", result)
    result = start.train_model(xgboost.XGBClassifier(), start.xtrain_tfidf, start.train_y, start.xvalid_tfidf)
    print("xgboost, WordLevel TF-IDF: ", result)
    print('-------------------------------------------------------------------------------------------------------')


