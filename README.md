# Heart-Disease-Detection
Implemented the K-Nearest Neighbour (KNN) Model for Binary Classification from scratch, to detect if a given patient would be susceptible to Heart Disease.

# Description
Developed a KNN Model for Binary Classification (from the ground up, without using existing libraries). The Model implements various L1, L2 and cosine distance measures to obtain similarity between two points from the training data.

The 'K' value for any dataset (in this case, Heart Disease Dataset from the UCI Machine Learning Repository) can be learnt by tuning HyperParameters of the model, both for Normalized and Scaled KNN and a vanilla model of KNN.

This code can be used with any dataset, by modifying the code and dataset in the data.py file.

# Model Performance
    **Without Scaling**
    best 'k' = 11
    best distance function = gaussian
    f1_score for current configuration= 0.7727272727272727
    
    **With Scaling**
    best 'k' = 5
    best distance function = euclidean
    best scaler = min_max_scale
    f1_score for current configuration= 0.8888888888888888
