from data import data_processing
from utils import Distances, HyperparameterTuner, NormalizationScaler, MinMaxScaler, f1_score


def main():
    distance_funcs = {
        'euclidean': Distances.euclidean_distance,
        'minkowski': Distances.minkowski_distance,
        'gaussian': Distances.gaussian_kernel_distance,
        'inner_prod': Distances.inner_product_distance,
        'cosine_dist': Distances.cosine_similarity_distance,
    }

    scaling_classes = {
        'min_max_scale': MinMaxScaler,
        'normalize': NormalizationScaler,
    }

    x_train, y_train, x_val, y_val, x_test, y_test = data_processing()

    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)
    print('x_val shape = ', x_val.shape)
    print('y_val shape = ', y_val.shape)
    print('x_test shape = ', x_test.shape)
    print('y_test shape = ', y_test.shape)

    tuner_without_scaling_obj = HyperparameterTuner()
    tuner_without_scaling_obj.tuning_without_scaling(distance_funcs, x_train, y_train, x_val, y_val)

    print("**Without Scaling**")
    print("k =", tuner_without_scaling_obj.best_k)
    print("distance function =", tuner_without_scaling_obj.best_distance_function)
    print("f1_score=", tuner_without_scaling_obj.best_f1_score)
    pred = tuner_without_scaling_obj.best_model.predict(x_test)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            correct += 1
    accuracy = float(correct)/len(pred)
    print ("Accuracy is: ", accuracy)
    print ("F1 Score: ", f1_score(y_test, pred))

    tuner_with_scaling_obj = HyperparameterTuner()
    tuner_with_scaling_obj.tuning_with_scaling(distance_funcs, scaling_classes, x_train, y_train, x_val, y_val)
#
    print("\n**With Scaling**")
    print("k =", tuner_with_scaling_obj.best_k)
    print("distance function =", tuner_with_scaling_obj.best_distance_function)
    print("scaler =", tuner_with_scaling_obj.best_scaler)
    print("f1_score=", tuner_with_scaling_obj.best_f1_score)
    pred_2 = tuner_with_scaling_obj.best_model.predict(x_test)
    correct_2 = 0
    for i in range(len(pred_2)):
        if pred_2[i] == y_test[i]:
            correct_2 += 1
    accuracy_2 = float(correct_2)/len(pred_2)
    print ("Accuracy is: ", accuracy_2)
    print ("F1 Score:", f1_score(y_test, pred_2))


if __name__ == '__main__':
    main()


