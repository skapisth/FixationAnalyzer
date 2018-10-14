import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


class DNNClassifier(object):

    def __init__(self,
                     train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     lr,
                     num_classes,
                     batch_size,
                     loss = 'sparse_categorical_crossentropy',
                     decay = 1e-6,
                     momentum = 0.9,
                     nesterov = True,
                     activation_init = 'relu',
                     activation_final = 'softmax',
                     dropout = 0.5,
                     n_epochs = 10,
                     ):

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.loss = loss
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.activation_init = activation_init
        self.activation_final = activation_final
        self.dropout = dropout
        self.num_classes = len(np.unique(self.train_labels))
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model_build()
        self.compile_model()
        self.accuracy,self.predictions = self.fit_evalaute_predict()


    def model_build(self):
        self.model = Sequential()
        self.model.add(Dense(512, activation=self.activation_init, input_dim=self.train_data.shape[1]))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation=self.activation_init))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, self.activation=activation_final))
        return self.model

    def compile_model(self):
        self.sgd = SGD(lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=True)
        self.model.compile(loss=self.loss,optimizer=sgd,metrics=['accuracy'])
        return self.model


    def fit_evaluate_predict(self):
        self.model.fit(x_train, y_train,epochs=20,batch_size=128)
        score = self.model.evaluate(self.test_data, self.test_labels, batch_size=self.batch_size)
        predictions = self.model.predict(self.train_data,batch_size=self.batch_size,verbose=0)
        return score, predictions
