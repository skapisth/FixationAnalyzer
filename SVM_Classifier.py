from sklearn import svm
import numpy as np

class SVM_Classifier(object):

    def __init__(self,
                    C,
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    decision_function_shape,
                    kernel_type='rbf'):
        self.C = C
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.decision_function_shape = decision_function_shape
        self.kernel_type = kernel_type

        self.fit = self.__fit(self.train_data,self.train_labels)
        self.accuracy = self.__test(self.fit,self.test_data,self.test_labels)

    def __fit(self,train_data,train_labels):
        clfr = svm.SVC()
        classification_fit = clfr.fit(train_data,train_labels)
        return classification_fit

    def __test(self,fitted_clfr,test_data,test_labels):
        predicted_labels = fitted_clfr.predict(test_data)
        test_labels = np.array(test_labels)

        correct_predictions = (predicted_labels==test_labels)
        accuracy = np.sum(correct_predictions)/len(correct_predictions)

        return accuracy
