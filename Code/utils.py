import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    numerator = 2*sum([a*b for a, b in zip(real_labels, predicted_labels)])
    denominator = sum([a+b for a, b in zip(real_labels, predicted_labels)])
    f1score = numerator/denominator
    
    return f1score
    
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        minkowski_dist = (sum([abs((a-b))**3 for a, b in zip(point1, point2)]))**(float(1/3))
        return minkowski_dist
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        euclid_dist = (sum([(a-b)**2 for a, b in zip(point1, point2)]))**.5
        return euclid_dist
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        inner_product_dist = sum([(a*b) for a, b in zip(point1, point2)])
        return inner_product_dist
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        numerator = sum([(a*b) for a, b in zip(point1, point2)])
        denominator_1 = (sum([(a*b) for a, b in zip(point1, point1)]))**.5
        denominator_2 = (sum([(a*b) for a, b in zip(point2, point2)]))**.5
        cosine_similarity = numerator/(denominator_1*denominator_2)
        cosine_dist = 1 - cosine_similarity
        return cosine_dist
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        eu_dist = sum([(a-b)**2 for a, b in zip(point1, point2)])
        gaussian_kernel_dist = -np.exp(-1/2 * eu_dist)
        return gaussian_kernel_dist
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        K = range(1, 30, 2)
        tie_breaks = ['euclidean', 'minkowski', 'gaussian', 'inner_prod', 'cosine_dist' ]
        best_dist_func_test = None
        best_f1score = None
        best_model_test = None
        for k in K:
        #    print "K is: ", k
            
            for dist_func in distance_funcs:
        #        print "Dist_Func is: ", dist_func
                model = KNN(k, distance_funcs[dist_func])
        #                print "model instance:", zbc
                model.train(x_train, y_train)
                ypred = model.predict(x_val)
                f1score = f1_score(y_val, ypred)
        #        print "F1 Score is: ", f1score
        #                print dist_func, f1score, best_f1score
                if best_f1score == None:
                    best_f1score = f1score
                    best_k_test = k
                    best_dist_func_test = dist_func
                    best_model_test = model
                elif best_f1score < f1score:
                    best_f1score = f1score
                    best_k_test = k
                    best_dist_func_test = dist_func
                    best_model_test = model
                elif best_f1score == f1score:
#                    print "I am here"
                    index_dist_func = tie_breaks.index(dist_func)
                    index_best_dist_func = tie_breaks.index(best_dist_func_test)
#                    print index_dist_func, index_best_dist_func, best_k_test, k, f1score, best_f1score
                    if index_dist_func < index_best_dist_func:
                        best_dist_func_test = dist_func
                        best_k_test = k
                        best_model_test = model
#                        print "broken tie on Dist_func"
                    elif index_dist_func == index_best_dist_func:
                        if best_k_test > k:
                            best_k_test = k
                            best_model_test = model
#                            print "broken tie on smaller k"
        
        # You need to assign the final values to these variables
        self.best_k = best_k_test
        self.best_distance_function = best_dist_func_test
        self.best_model = best_model_test
        self.best_f1_score = best_f1score
        return
        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        K = range(1, 30, 2)
        tie_breaks = ['euclidean', 'minkowski', 'gaussian', 'inner_prod', 'cosine_dist' ]
        scalers = ['min_max_scale', 'normalize']
        best_k_test = None
        best_dist_func_test = None
        best_f1score = None
        best_model_test = None
        best_scaler_test = None
        for k in K:
#            print "K is: ", k
            for dist_func in distance_funcs:
#                print "Dist_Func is: ", dist_func
                zbc = KNN(k, distance_funcs[dist_func])
#                print "model instance:", zbc
                for scaler in scaling_classes:
#                    print "Scaler Model is: ", scaler
                    train_scaler = scaling_classes[scaler]()
                    train_op = train_scaler(x_train)
#                    validation_scaler = scaling_classes[scaler]()
                    validation_op = train_scaler(x_val)
                    zbc.train(train_op, y_train)
                    ypred = zbc.predict(validation_op)
                    f1score = f1_score(y_val, ypred)
#                    print "F1 Score is: ", f1score
    #                print dist_func, f1score, best_f1score
                    if best_f1score == None:
                        best_f1score = f1score
                        best_k_test = k
                        best_dist_func_test = dist_func
                        best_model_test = zbc
                        best_scaler_test = scaler
                    elif best_f1score < f1score:
                        best_f1score = f1score
                        best_k_test = k
                        best_dist_func_test = dist_func
                        best_model_test = zbc
                        best_scaler_test = scaler
                    elif best_f1score == f1score:
    #                    print "I am here"
                        index_best_scaler = scalers.index(best_scaler_test)
                        index_scaler = scalers.index(scaler)
                        if index_scaler < index_best_scaler:
                            best_scaler_test = scaler
                            best_dist_func_test = dist_func
                            best_k_test = k
                            best_model_test = zbc
                        elif index_best_scaler == index_scaler:
                            index_dist_func = tie_breaks.index(dist_func)
                            index_best_dist_func = tie_breaks.index(best_dist_func_test)
        #                    print index_dist_func, index_best_dist_func, best_k_test, k, f1score, best_f1score
                            if index_dist_func < index_best_dist_func:
                                best_dist_func_test = dist_func
                                best_k_test = k
                                best_model_test = zbc
        #                        print "broken tie on Dist_func"
                            elif index_dist_func == index_best_dist_func:
                                if best_k_test > k:
                                    best_k_test = k
                                    best_model_test = zbc
    #                            print "broken tie on smaller k"
        
        # You need to assign the final values to these variables
        self.best_k = best_k_test
        self.best_distance_function = best_dist_func_test
        self.best_scaler = best_scaler_test
        self.best_model = best_model_test
        self.best_f1_score = best_f1score
        return
        raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized_features = []
        for point in features:
            normalized_point = []
            denominator = (sum(i**2 for i in point))**.5
            for x in point:
                if x == 0:
                    normalized_point.append(0)
                else:
                    normalized_x = x/denominator
                    normalized_point.append(normalized_x)
            normalized_features.append(normalized_point)

        return normalized_features
        raise NotImplementedError


class MinMaxScaler:
    
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """
    
    def __init__(self):
        pass
        self.first_run = True
        self.max_points = None
        self.min_points = None
    
    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
#        global first_run
#        global max_points
#        global min_points
        if self.first_run:
#            print "First Run"
            features_np = np.asarray(features)
            features_np = np.transpose(features_np)
            self.max_points = [max(features_np[x]) for x in range(len(features_np))]
            self.min_points = [min(features_np[x]) for x in range(len(features_np))]
            self.first_run = False
            min_max_normalized_features = []
            for point in features:
                min_max_normalized_points = []
                for i in range(len(point)):
#                    print i
                    denominator = self.max_points[i] - self.min_points[i]
                    if denominator == 0:
                        min_max_normalized_points.append(0)
                    else:
                        min_max_normalized_x = round((point[i] - self.min_points[i])/float(denominator), 6)
                        min_max_normalized_points.append(min_max_normalized_x)
                min_max_normalized_features.append(min_max_normalized_points)
        
            return min_max_normalized_features
        else:    
            if self.max_points == None or self.min_points == None:
                features_np = np.asarray(features)
                features_np = np.transpose(features_np)
                self.max_points = [max(features_np[x]) for x in range(len(features_np))]
                self.min_points = [min(features_np[x]) for x in range(len(features_np))]
#            print "Not 1st Run"
            min_max_normalized_features = []
#            features_np = np.asarray(features)
#            features_np = np.transpose(features_np)
#            max_points = [max(features_np[x]) for x in range(len(features_np))]
#            min_points = [min(features_np[x]) for x in range(len(features_np))]
            for point in features:
#                print point
                min_max_normalized_points = []
                for i in range(len(point)):
#                    print i
                    denominator = self.max_points[i] - self.min_points[i]
                    if denominator == 0:
                        min_max_normalized_points.append(0)
                    else:
                        min_max_normalized_x = round((point[i] - self.min_points[i])/float(denominator), 6)
                        min_max_normalized_points.append(min_max_normalized_x)
                min_max_normalized_features.append(min_max_normalized_points)
        
            return min_max_normalized_features
        raise NotImplementedError
