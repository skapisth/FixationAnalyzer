from importlib import import_module
import cv2
import numpy as np


class FeatureExtractor(object):
    """
    Class to extract features from pretrained neural networks
    trained on imagenet with max pooling applied.

    This class utilizes keras to automatically leverage
    hardware resources and retrieve pretrained networks.
    available networks are:
                            - xception
                            - vgg16
                            - vgg19
                            - resnet50
                            - inception_v3
                            - inception_resnet_v2
                            - mobilenet
                            - densenet121
                            - densenet169
                            - densenet201
                            - nasnetlarge
                            - nasnetmobile
                            - mobilenetv2

    see: https://keras.io/applications/ for more details
    kwargs for network instantiation are:
                                        include_top=False,
                                        weights='imagenet',
                                        pooling='avg'


    Instantiation Args:
        network_name (str): name of network to extract features from
        interpolation (cv2 constant): type of interpolation used to
                                        resize images

    Example Use Case:
        network = FeatureExtractor('resnet50')

        lenna_features = network.extract_features('lenna')

    """
    __SUBMODULES = {'xception': 'keras.applications.xception',
                    'vgg16': 'keras.applications.vgg16',
                    'vgg19': 'keras.applications.vgg19',
                    'resnet50': 'keras.applications.resnet50',
                    'inception_v3': 'keras.applications.inception_v3',
                    'inception_resnet_v2': 'keras.applications.inception_resnet_v2',
                    'mobilenet': 'keras.applications.mobilenet',
                    'densenet121': 'keras.applications.densenet',
                    'densenet169': 'keras.applications.densenet',
                    'densenet201': 'keras.applications.densenet',
                    'nasnetlarge': 'keras.applications.nasnet',
                    'nasnetmobile': 'keras.applications.nasnet',
                    'mobilenetv2': 'keras.applications.mobilenetv2',
                    }
    __FUNCTION_NAMES = {'xception': 'Xception',
                        'vgg16': 'VGG16',
                        'vgg19': 'VGG19',
                        'resnet50': 'ResNet50',
                        'inception_v3': 'InceptionV3',
                        'inception_resnet_v2': 'InceptionResNetV2',
                        'mobilenet': 'MobileNet',
                        'densenet121': 'DenseNet121',
                        'densenet169': 'DenseNet169',
                        'densenet201': 'DenseNet201',
                        'nasnetlarge': 'NASNetLarge',
                        'nasnetmobile': 'NASNetMobile',
                        'mobilenetv2': 'MobileNetV2',
                        }

    def __init__(self,
                 network_name='inception_v3',
                 pooling_type='avg'):

        self.model, self.preprocess_fn, self.kerasbackend\
            = self.__keras_importer(network_name,pooling_type)
        self.network_name = network_name
        self.pooling_type = pooling_type

    def extract_features_from_specific_layer(self, imgs, layer_idx):
        """
        Extracts image features from a specific layer the neural network specified in
        __init__

        input::
            imgs (list): list of images
            layer_idx: layer from where you want features
        returns::
            features (np.ndarray): features for this image
        """
        imgs = self.__build_image_data(imgs)
        get_features = self.kerasbackend.function([self.model.layers[0].input,
                                                    self.kerasbackend.learning_phase()],
                                                    [self.model.layers[layer_idx].output])
        features = get_features([imgs,0])

        batch_features = np.vsplit(features[0],features[0].shape[0])
        features = np.vstack( [arr.reshape(1,arr.size) for arr in batch_features] )
        return features


    def extract_features(self, imgs):
        """
        Extracts image features from a the neural network specified in
        __init__

        input::
            imgs (list): list of images
        returns::
            features (np.ndarray): features for this image
        """
        # Error checking for img occurs in __build_image_data
        imgs = self.__build_image_data(imgs)
        features = self.model.predict(imgs)
        return features

    def __build_image_data(self, imgs):
        """
        this function turns an input numpy array or image path into an
        array format which keras requires for network feeding
        that format being a 4D tensor (batch,rows,cols,bands)
        (batch size will be always be 1 in this case)

        input::
            img (list):
                    list of numpy arrays
            preprocess_fn (func):
                    preprocessing function for image data, output from
                    self.__keras_importer
        returns::
            img_data (np.ndarray):
                    4D numpy array of the form (num_images,rows,cols,bands)
        """
        # the image must be numpy array so it can be processed
        if not isinstance(imgs, list):
            raise ValueError("imgs must be a list of numpy arrays.")

        imgs = np.stack(imgs,axis=0)

        # preprocessing the image
        img_data = self.preprocess_fn(imgs)
        return img_data

    def __keras_importer(self, network_name, pooling_type):
        """
        Retrieves the feature extraction algorithm and preprocess_fns
        from keras, only importing the desired model specified by the
        network name.

        input::
            network_name (str):
                name of the network being used for feature extraction
            pooling_type (str):
                type of pooling you want to use ('avg' or 'max')

        returns::
            1) model (func):
                function that extract features from the NN
            2) preprocess_fn (func):
                function that preprocesses the image for the network
        """
        # checking to make sure network_name is valid
        if network_name not in self.__SUBMODULES:
            error_string = "unknown network '{network_name}', \
                            network_name must be one of {network_list}"\
                                .format(network_name=network_name,
                                        network_list=self.__SUBMODULES.keys())
            raise ValueError(error_string)

        assert pooling_type in ['avg','max'],"'pooling_type' must be one of the following strings ['avg','max']"

        # importing the proper keras model and preprocess_fn
        submodule = import_module(self.__SUBMODULES[network_name])
        model_constructor = getattr(submodule,
                                    self.__FUNCTION_NAMES[network_name])
        model = model_constructor(include_top=False,
                                     weights='imagenet',
                                     pooling=pooling_type)

        preprocess_fn = getattr(submodule, 'preprocess_input')


        from keras import backend as kerasbackend

        return model, preprocess_fn, kerasbackend

    def __del__(self):
        del self.model
        self.kerasbackend.clear_session()



# END
