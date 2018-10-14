import cv2
import os
import math
import numpy as np
import fixationanalyzer as fa
from fixationanalyzer import FeatureExtractor
from fixationanalyzer import FisherVectorExtraction
from fixationanalyzer import SVM_Classifier
from sklearn.decomposition import PCA
from datetime import datetime
import tensorflow as tf
import time


CONFIGS = {}
# CONFGIS["FIXATION_FILENAME"] = './clean_data.csv'
# CONFGIS["TRAINING_FRACTION"] = .7
# CONFGIS["K_FOLDS"] = 10
# CONFGIS["CLASSIFICATION_TYPE"] = 'exp_v_nov'
# CONFGIS["IMAGE_DIR"] = './images'
# CONFGIS["ALGORITHM_TYPE"] = 'mfpa-cnn(L)'
# CONFGIS["PATCH_LENGTH"] = 9
# CONFGIS["BATCH_SIZE"] = 60
# CONFGIS["NETWORK_NAME"] = 'resnet50'
# CONFGIS["POOLING_TYPE"] = 'avg'
# CONFGIS["LAYER_IDX"] = 2
# CONFGIS["N_KERNELS"] = 5
# CONFGIS["COVARIANCE_TYPE"] = 'diag'
# CONFGIS["REG_COVAR"] = 1e-5
# CONFGIS["N_COMPONENTS_FOR_PCA"] = 256
# CONFGIS["KERNEL_TYPE"] = 'rbf'
# CONFGIS["C"] = 1
# CONFGIS["DECISION_FUNCTION_TYPE"] = 'ovr' if CONFGIS["CLASSIFICATION_TYPE"] != 'four' else 'ovo'

IMG_CACHE = {}
def get_image_patches(fixations,labels):
    """
    Generator which returns batches of image patches of specified size

    """

    patch_batch = []
    label_batch = []
    batch_index = 1
    for paintings,label in zip(fixations,labels):
        for painting_id,painting_fixations in paintings.items():
            #adding the image to image cache if it's not already there
            if painting_id not in IMG_CACHE:
                image_name = os.path.join(CONFIGS['IMAGE_DIR'],str(painting_id)+'.jpg')
                IMG_CACHE[painting_id] = cv2.imread(image_name)

            painting_img = IMG_CACHE[painting_id]
            height,width = painting_img.shape[:2]
            max_left = math.ceil(CONFIGS['PATCH_LENGTH']/2)
            max_right = width - math.ceil(CONFIGS['PATCH_LENGTH']/2)
            max_top = math.ceil(CONFIGS['PATCH_LENGTH']/2)
            max_bottom = height - math.ceil(CONFIGS['PATCH_LENGTH']/2)

            for i,fix in enumerate(painting_fixations):
                #throwing away fixation data that is close to the image border
                X,Y = fix['X'],fix['Y']
                if X<max_left:
                    continue
                if X>max_right:
                    continue
                if Y<max_top:
                    continue
                if Y>max_bottom:
                    continue
                left = X - (CONFIGS["PATCH_LENGTH"] // 2)
                right = X + (CONFIGS["PATCH_LENGTH"] - CONFIGS["PATCH_LENGTH"]//2)
                top = Y - (CONFIGS["PATCH_LENGTH"] // 2)
                bottom = Y + (CONFIGS["PATCH_LENGTH"] - CONFIGS["PATCH_LENGTH"]//2)
                patch = painting_img[top:bottom,left:right,:].astype(np.float32)
                patch_batch.append( patch )
                label_batch.append( label )

                if len(patch_batch) == CONFIGS["BATCH_SIZE"]:
                    # fa.info("Batch index of {0} of image patches yielded.".format(batch_index))
                    # fa.debug("length of batch:{0}.".format(len(patch_batch)))
                    batch_index+=1
                    yield (patch_batch,label_batch)
                    patch_batch = []
                    label_batch = []

    if len(patch_batch) > 0:
        # fa.info("Batch index of {0} of image patches yielded.".format(batch_index))
        batch_index+=1
        yield (patch_batch,label_batch)


def extract_features(fixations,labels):

    patch_gen = get_image_patches(fixations,labels)

    #batches, label_batches = get_image_patches(fixations,labels)
    feature_extractor = FeatureExtractor(CONFIGS["NETWORK_NAME"],CONFIGS["POOLING_TYPE"])

    all_features = []
    all_labels = []
    batch_index = 1
    # with tf.device("/gpu:0"):
    for batch,label_batch in patch_gen:
        if CONFIGS['ALGORITHM_TYPE'] == 'mfpa-cnn(L)':
            features = feature_extractor.extract_features_from_specific_layer(batch,CONFIGS["LAYER_IDX"])
        elif CONFIGS['ALGORITHM_TYPE'] == 'mfpa-cnn(H)':
            features = feature_extractor.extract_features(batch)
        batch_index+=1
        all_features.append(features)
        all_labels.extend(label_batch)

    fa.info("all neural net features extracted for algorithm type {0}".format(CONFIGS['ALGORITHM_TYPE']))

    all_features = np.vstack(all_features)

    return all_features,all_labels


def format_fixations(fixations):
    pass

def test(configs):
    global CONFIGS
    CONFIGS = configs
    CONFIGS['FIXATION_FILENAME'] = str(CONFIGS['FIXATION_FILENAME'])
    CONFIGS['TRAINING_FRACTION'] = float(CONFIGS['TRAINING_FRACTION'])
    CONFIGS['K_FOLDS'] = int(CONFIGS['K_FOLDS'])
    CONFIGS['CLASSIFICATION_TYPE'] = str(CONFIGS['CLASSIFICATION_TYPE'])
    CONFIGS['IMAGE_DIR'] = str(CONFIGS['IMAGE_DIR'])
    CONFIGS['ALGORITHM_TYPE'] = str(CONFIGS['ALGORITHM_TYPE'])
    CONFIGS['PATCH_LENGTH'] = int(CONFIGS['PATCH_LENGTH'])
    CONFIGS['BATCH_SIZE'] = int(CONFIGS['BATCH_SIZE'])
    CONFIGS['NETWORK_NAME'] = str(CONFIGS['NETWORK_NAME'])
    CONFIGS['POOLING_TYPE'] = str(CONFIGS['POOLING_TYPE'])
    CONFIGS['LAYER_IDX'] = int(CONFIGS['LAYER_IDX'])
    CONFIGS['N_KERNELS'] = int(CONFIGS['N_KERNELS'])
    CONFIGS['COVARIANCE_TYPE'] = str(CONFIGS['COVARIANCE_TYPE'])
    CONFIGS['REG_COVAR'] = float(CONFIGS['REG_COVAR'])
    CONFIGS['N_COMPONENTS_FOR_PCA'] = int(CONFIGS['N_COMPONENTS_FOR_PCA'])
    CONFIGS['KERNEL_TYPE'] = str(CONFIGS['KERNEL_TYPE'])
    CONFIGS['C'] = float(CONFIGS['C'])
    CONFIGS['LEARNING_RATE'] = float(CONFIGS['LEARNING_RATE'])


    start_datetime = datetime.now()
    dataset_manager = fa.DatasetManager(CONFIGS["FIXATION_FILENAME"],
                                        training_fraction=CONFIGS["TRAINING_FRACTION"],
                                        num_chunks=CONFIGS["K_FOLDS"],
                                        classification_type=CONFIGS["CLASSIFICATION_TYPE"])

    accuracies_k_folds = []
    for k in range(CONFIGS["K_FOLDS"]):
        k += 1
        train_fixations,train_labels,test_fixations,test_labels = dataset_manager.get_train_test()


        if CONFIGS["ALGORITHM_TYPE"] in ['mfpa-cnn(L)','mfpa-cnn(H)']:
            #generate features on data

            train_features,train_labels = extract_features(train_fixations,train_labels)
            train_features = np.expand_dims(train_features,axis=1)
            test_features,test_labels = extract_features(test_fixations,test_labels)
            test_features = np.expand_dims(test_features,axis=1)

            fisher_vector_extraction = FisherVectorExtraction(CONFIGS['N_KERNELS'], CONFIGS['COVARIANCE_TYPE'], CONFIGS['REG_COVAR'])
            fa.info("Fitting GMM on train set. ",'(fold={},permutation={})'.format(k,CONFIGS['PERM_IDX']))
            fisher_vector_extraction.fit(train_features)
            fa.info("Predicting Fisher Vectors on Train set. ",'(fold={},permutation={})'.format(k,CONFIGS['PERM_IDX']))
            train_fisher_vectors = fisher_vector_extraction.predict(train_features)
            fa.info("Predicting Fisher Vectors on Test set. ",'(fold={},permutation={})'.format(k,CONFIGS['PERM_IDX']))
            test_fisher_vectors = fisher_vector_extraction.predict(test_features)

            #Reducing dimensions with pCA
            pca = PCA(n_components=CONFIGS['N_COMPONENTS_FOR_PCA'])
            pca.fit(train_fisher_vectors)
            train_fisher_vectors = pca.transform(train_fisher_vectors)
            test_fisher_vectors = pca.transform(test_fisher_vectors)
            #Classification
            fa.info("Entering classification mode. ",'(fold={}/{},permutation={}/{})'.format(k,CONFIGS['K_FOLDS'],CONFIGS['PERM_IDX'],CONFIGS['NUM_PERM']))

            # classifier = SVM_Classifier(CONFIGS["C"],
            #                             train_fisher_vectors,
            #                             train_labels,
            #                             test_fisher_vectors,
            #                             test_labels,
            #                             CONFIGS['DECISION_FUNCTION_TYPE'],
            #                             CONFIGS['KERNEL_TYPE'])
            # accuracy = classifier.accuracy

            classifier = DNNClassifier(train_fisher_vectors,
                                       train_labels,
                                       test_fisher_vectors,
                                       test_labels,
                                       CONFIGS['LEARNING_RATE'],
                                       num_classes,
                                       CONFIGS['BATCH_SIZE'],
                                       loss=loss,
                                       decay = decay,
                                       momentum = momentum,
                                       nesterov = nesterov,
                                       activation_init = activation_init,
                                       activation_final = activation_final,
                                       dropout = dropout,
                                       n_epochs = n_epochs)
            accuracy = classifier.accuracy
            accuracies_k_folds.append(accuracy)

        dataset_manager.rotate()

    k_fold_classification_accuracy_with_mean = np.mean(accuracies_k_folds)
    k_fold_classification_accuracy_with_std = np.std(accuracies_k_folds)*2

    print("classification accuracy {0}% for algorithm_type {1} with 95% CI {2}.".format(k_fold_classification_accuracy_with_mean*100,
                                                                                      CONFIGS['ALGORITHM_TYPE'],
                                                                                      k_fold_classification_accuracy_with_std))

    with open('results_{}.txt'.format(start_datetime),'w') as f:
        f.write("START_DATETIME: {}".format(start_datetime))
        f.write("\n")
        f.write("FIXATION_FILENAME: {}".format(CONFIGS['FIXATION_FILENAME']))
        f.write("\n")
        f.write("TRAINING_FRACTION: {}".format(CONFIGS['TRAINING_FRACTION']))
        f.write("\n")
        f.write("K_FOLDS: {}".format(CONFIGS['K_FOLDS']))
        f.write("\n")
        f.write("CLASSIFICATION_TYPE: {}".format(CONFIGS['CLASSIFICATION_TYPE']))
        f.write("\n")
        f.write("IMAGE_DIR: {}".format(CONFIGS['IMAGE_DIR']))
        f.write("\n")
        f.write("ALGORITHM_TYPE: {}".format(CONFIGS['ALGORITHM_TYPE']))
        f.write("\n")
        f.write("PATCH_LENGTH: {}".format(CONFIGS['PATCH_LENGTH']))
        f.write("\n")
        f.write("BATCH_SIZE: {}".format(CONFIGS['BATCH_SIZE']))
        f.write("\n")
        f.write("NETWORK_NAME: {}".format(CONFIGS['NETWORK_NAME']))
        f.write("\n")
        f.write("POOLING_TYPE: {}".format(CONFIGS['POOLING_TYPE']))
        f.write("\n")
        f.write("LAYER_IDX: {}".format(CONFIGS['LAYER_IDX']))
        f.write("\n")
        f.write("N_KERNELS: {}".format(CONFIGS['N_KERNELS']))
        f.write("\n")
        f.write("COVARIANCE_TYPE: {}".format(CONFIGS['COVARIANCE_TYPE']))
        f.write("\n")
        f.write("REG_COVAR: {}".format(CONFIGS['REG_COVAR']))
        f.write("\n")
        f.write("N_COMPONENTS_FOR_PCA: {}".format(CONFIGS['N_COMPONENTS_FOR_PCA']))
        f.write("\n")
        f.write("KERNEL_TYPE: {}".format(CONFIGS['KERNEL_TYPE']))
        f.write("\n")
        f.write("C: {}".format(CONFIGS['C']))
        f.write("\n")
        f.write("ACCURACY:{}".format(k_fold_classification_accuracy_with_mean))
        f.write("\n")
        f.write("95CI:{}".format(k_fold_classification_accuracy_with_std))
        f.write("\n")

    CONFIGS = {}

if __name__ == "__main__":
    fa.main()
