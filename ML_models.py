#!/usr/bin/python
# 
# Various ML models from scikit-learn
#
# includes:
# LogisticRegression
# SVM
# KNN
# NaiveBayes
# Perceptron (1-layer NN)
# Linear SVC
# SGD
# Decision Tree
# Random Forest
#
# Author: Chaney Lin
# Date: April 2018
#
import pandas as pd

def runModel(model, X_train, Y_train, X_test):
    """
    performs fitting of [model] using [X_train] and [Y_train]
    returns accuracy on training set, and predictions on [X_test]
    """
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = round(model.score(X_train, Y_train) * 100, 2)
    return accuracy, Y_pred

def runLogisticRegression(X_train, Y_train, X_test):
    """
    runs logistic regression
    returns accuracy on training set, and predictions
    """
    from sklearn.linear_model import LogisticRegression
    
    logreg = LogisticRegression()
    return runModel(logreg, X_train, Y_train, X_test)

def runSVM(X_train, Y_train, X_test):
    """
    runs support vector machine
    returns accuracy on training set, and predictions
    """
    from sklearn.svm import SVC
    
    svc = SVC()
    return runModel(svc, X_train, Y_train, X_test)

def runKNN(X_train, Y_train, X_test, n_neighbors = 3):
    """
    runs K-nearest neighbors (input K)
    returns accuracy on training set, and predictions
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier(n_neighbors = n_neighbors)
    return runModel(knn, X_train, Y_train, X_test)

def runNaiveBayes(X_train, Y_train, X_test):
    """
    runs Naive Bayes
    returns accuracy on training set, and predictions
    """
    from sklearn.naive_bayes import GaussianNB
    
    gaussian = GaussianNB()
    return runModel(gaussian, X_train, Y_train, X_test)

def runPerceptron(X_train, Y_train, X_test):
    """
    runs Perceptron (i.e. single layer NN)
    returns accuracy on training set, and predictions
    """

    from sklearn.linear_model import Perceptron

    perceptron = Perceptron()
    return runModel(perceptron, X_train, Y_train, X_test)

def runLinearSVC(X_train, Y_train, X_test):
    """
    runs linear SVC
    returns accuracy on training set, and predictions
    """

    from sklearn.svm import SVC, LinearSVC

    linear_svc = LinearSVC()
    return runModel(linear_svc, X_train, Y_train, X_test)

def runSGD(X_train, Y_train, X_test):
    """
    runs stochastic gradient descent
    returns accuracy on training set, and predictions
    """
    from sklearn.linear_model import SGDClassifier

    sgd = SGDClassifier()
    return runModel(sgd, X_train, Y_train, X_test)

def runDecisionTree(X_train, Y_train, X_test):
    """
    runs decision tree
    returns accuracy on training set, and predictions
    """

    from sklearn.tree import DecisionTreeClassifier

    decision_tree = DecisionTreeClassifier()
    return runModel(decision_tree, X_train, Y_train, X_test)

def runRandomForest(X_train, Y_train, X_test, n_estimators = 300):
    """
    runs random forest with [n_estimators] estimators
    returns accuracy on training set, and predictions
    """

    from sklearn.ensemble import RandomForestClassifier

    random_forest = RandomForestClassifier(n_estimators=300)
    return runModel(random_forest, X_train, Y_train, X_test)

def runAll(X_train, Y_train, X_test, n_neighbors = 3, n_estimators = 300):
    """
    runs all models
    optional inputs are number of neighbors (for KNN) and number of estimators (for random forest)
    """

    modelnames = ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                  'Random Forest', 'Naive Bayes', 'Perceptron', 
                  'Stochastic Gradient Decent', 'Linear SVC', 
                  'Decision Tree']

    acc = []
    pred = []

    svm = runSVM(X_train, Y_train, X_test)
    knn = runKNN(X_train, Y_train, X_test, n_neighbors = n_neighbors)
    logreg = runLogisticRegression(X_train, Y_train, X_test)
    random_forest = runRandomForest(X_train, Y_train, X_test, n_estimators = n_estimators)
    naive_bayes = runNaiveBayes(X_train, Y_train, X_test)
    perceptron = runPerceptron(X_train, Y_train, X_test)
    sgd = runSGD(X_train, Y_train, X_test)
    linear_svc = runLinearSVC(X_train, Y_train, X_test)
    decision_tree = runDecisionTree(X_train, Y_train, X_test)

    for results in [svm, knn, logreg, random_forest, naive_bayes, perceptron, sgd, linear_svc, decision_tree]:
        acc.append(results[0])
        pred.append(results[1])

    models = pd.DataFrame({
        'Model': modelnames,
        'Score': acc,
        'Predictions': pred})

    return models.sort_values(by='Score', ascending=False)