import joblib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score,classification_report
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import librosa
import warnings
warnings.filterwarnings('ignore')

scaler = StandardScaler()

random_seed = 42
np.random.seed(random_seed)

def add_more_features_using_lag(df_shift,top_cols):    
    for col in top_cols:
        df_shift[col+"_rolling_mean"] = df_shift[col].rolling(window=25).mean()
        df_shift[col+"_rolling_std"] = df_shift[col].rolling(window=25).std()
        df_shift[col+"_ema"] = df_shift[col].ewm(span=25, adjust=False).mean()
    return df_shift

         
def knn_classifier(X_train_scaled, y_train,X_test_scaled,y_test):
    # Create an instance of KNeighborsClassifier
    knn = KNeighborsClassifier()

    # # Parameter grid for K-Nearest Neighbors
    # # param_grid = {'n_neighbors': [3, 5, 7, 9],
    # #                   'weights': ['uniform', 'distance'],
    # #                   'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    # #              'p':[1,2]}


    param_grid = {'n_neighbors': [3],#3, 5, 7, 9],
                  'weights': [ 'distance'],#,'uniform'],
                  'algorithm': ['auto'],#, 'ball_tree', 'kd_tree', 'brute'],
                  'p': [1],#, 2],
                  'leaf_size': [10],#,, 20, 30],
                  'metric': ['euclidean'],#, 'manhattan', 'chebyshev'],
                  'metric_params': [None]}#, {'w': [0.1, 0.9]}]}


    # # param_grid = {'n_neighbors': [3],
    # #                   'weights': [ 'distance'],
    # #                   'algorithm': ['auto'],
    # #              'p':[1,2]}
    # # Best hyperparameters: {'algorithm': 'auto', 'n_neighbors': 3, 'weights': 'distance'}
    # # Best hyperparameters: {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'euclidean', 'metric_params': None, 
    # # 'n_neighbors': 9, 'p': 1, 'weights': 'distance'}
    
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # save
    best_knn = grid_search.best_estimator_
    # joblib.dump(best_knn, "C:/KNN_Classifier_Model.pkl")


    # Print best hyperparameters and score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    knn_preds = best_knn.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, knn_preds)
    print("KNN Accuracy:", knn_accuracy)

    print("Test accuracy score: ", accuracy_score(y_test,knn_preds)*100)
    print("Test f1 score: ", f1_score(y_test, knn_preds, average='weighted'))
    return knn_preds

    

def randomforest_classifier(X_train_scaled, y_train,X_test_scaled,y_test):
    # Create an instance of RandomForestClassifier
    rf_classifier = RandomForestClassifier()
    # param_grid = {
    #     'n_estimators': [50, 100, 150],
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False],
    #     'class_weight': [None, 'balanced', 'balanced_subsample'],
    #     'oob_score': [True, False],
    #     'random_state': [42]
    # }

    param_grid = {
        'n_estimators': [1500],
        'criterion': ['gini'],
        'max_depth': [10],
        'bootstrap': [True],
        'random_state': [42]
    }

    ## Best parameters: {'bootstrap': True, 'criterion': 'gini', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 100} 

    # # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
    
    # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
    # grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
    # grid_search = GridSearchCV(knn, param_grid, cv=20)#clf
    grid_search.fit(X_train_scaled, y_train)

    # save
    best_rf = grid_search.best_estimator_
    # joblib.dump(best_rf, "C:/RandomForest_classifier_Model.pkl")


    # Print best hyperparameters and score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    rf_preds = best_rf.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    print("RandomForest Accuracy:", rf_accuracy)

    print("Test accuracy score: ", accuracy_score(y_test,rf_preds)*100)
    print("Test f1 score: ", f1_score(y_test, rf_preds, average='weighted'))
    return rf_preds

#################################################################

def svm_classifier(X_train_scaled, y_train,X_test_scaled,y_test):

    # param_grid = {'C': [0.001,0.01,0.1, 1],
    #           'gamma': [0.001,0.01,0.1, 1,'scale', 'auto'],
    #           'kernel': ['linear', 'rbf', 'poly']
    #          }

    # after gridsearch the following are the best parameters
    param_grid = {'C':[1],
                'gamma': [0.10],
                'kernel': [ 'rbf'] 
                }

    # Train an SVM classifier with RBF kernel and tune hyperparameters using grid search
    clf = svm.SVC(kernel='rbf')

    # # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')    
    grid_search.fit(X_train_scaled, y_train)

    
    best_svm = grid_search.best_estimator_
    # save
    # joblib.dump(best_rf, "C:/SVM_classifier_Model.pkl")


    # Print best hyperparameters and score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)


    # Predict using SVM
    svm_preds = best_svm.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_preds)
    print("svm Accuracy:", svm_accuracy)

    print("Test accuracy score: ", accuracy_score(y_test,svm_preds)*100)
    print("Test f1 score: ", f1_score(y_test, svm_preds, average='weighted'))
    return svm_preds

def calculate_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test,y_pred))

    if cm.shape == (2, 2):
        TP=cm[0,0]
        TN=cm[1,1]
        FP=cm[0,1]
        FN=cm[1,0]
        print('Confusion matrix\n\n', cm)
        print('\nTrue Positives(TP) = ', cm[0,0])
        print('\nTrue Negatives(TN) = ', cm[1,1])
        print('\nFalse Positives(FP) = ', cm[0,1])
        print('\nFalse Negatives(FN) = ', cm[1,0])
        
        # Count the number of matching elements
        num_matches = sum(1 for gt, pred in zip(y_test, y_pred) if gt == pred)

        # Print the count of matches
        print("Number of matches:", num_matches, "total data: ", len(X_test))

        classification_error = (FP + FN) / float(TP + TN + FP + FN)
        print('Test Classification error : {0:0.4f}'.format(classification_error))

        precision = TP / float(TP + FP)
        print('Test Precision : {0:0.4f}'.format(precision))

        recall = TP / float(TP + FN)
        print('Test Recall or Sensitivity : {0:0.4f}'.format(recall))

        F1_score= 2*precision*recall/(precision+recall)
        print('Test F1_score from Precision and Recall : {0:0.4f}'.format(F1_score))
    print('--------------------------------------------------------------------')    

    
if __name__ == '__main__':
    
    filepath='C:/test_drone_Features/'

    df=pd.read_csv(filepath)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    top_cols = df.columns[:-1]
    X_all = df.loc[:, top_cols]

    X=add_more_features_using_lag(X_all,top_cols)
    column_means = X.mean()
    X = X.fillna(column_means) # features
    y = df.loc[:, 'label'] #target

    

    # Load  data and split it into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit on training data and transform both training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  
    knn_preds=knn_y_pred= knn_classifier(X_train_scaled, y_train,X_test_scaled,y_test)
    rf_preds=rf_y_pred = randomforest_classifier(X_train_scaled, y_train,X_test_scaled,y_test)
    svm_preds=svm_y_pred = svm_classifier(X_train_scaled, y_train,X_test_scaled,y_test)

    calculate_confusion_matrix(y_test, knn_preds)
    calculate_confusion_matrix(y_test, rf_preds)
    calculate_confusion_matrix(y_test, svm_preds)

########## Step:3 Train and Test Different Models ##########    
    