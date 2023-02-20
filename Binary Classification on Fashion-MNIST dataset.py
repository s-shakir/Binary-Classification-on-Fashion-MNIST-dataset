def load_data():

    Xtr=np.loadtxt("TrainData.csv")
    Ytr=np.loadtxt("TrainLabels.csv")
    Xts=np.loadtxt("TestData.csv")



    # fit the PCA with the training data
    pca = PCA().fit(Xtr)

    # Plot the cumulative sum of eigenvalues
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Features')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Fashion-MNIST Variance')
    plt.show()

    # The Variance graph showed that by using around 300 features, we can retain approximately >90% of the data.
    pca = PCA(n_components=300)
    pca.fit(Xtr)


    x_train_pca = pca.transform(Xtr)
    x_test_pca = pca.transform(Xts)


    return x_train_pca, x_test_pca, Ytr


def KNN_model(x_train_pca, Ytr):
    k = 5
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    model = KNeighborsClassifier(n_jobs=-1)

    params = {'n_neighbors':[6,8,10,12,16,18,20],
          'metric':['euclidean', 'manhattan','minkowski'],
          'algorithm':['brute', 'auto']
          }

    model1 = GridSearchCV(model, param_grid=params, n_jobs=1)

    acc_score = []

    fold = 1

    for train_index , test_index in kf.split(x_train_pca):
      X_train , X_test, y_train , y_test = x_train_pca[train_index],x_train_pca[test_index], Ytr[train_index] , Ytr[test_index]

      model1.fit(X_train,y_train)

      print("Best Hyper Parameters:",model1.best_params_)

      pred_values = model1.predict(X_test)

      acc = accuracy_score(pred_values , y_test)

      acc_score.append(acc)

      print("\nAccuracy of fold: {}".format(acc_score))

      fold += 1

    avg_acc_score = sum(acc_score)/k

    #print('accuracy of each fold: {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

def DT_model(x_train_pca, Ytr):
    k = 5
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    model = DecisionTreeClassifier(random_state=1234)

    params = {'criterion': ['gini', 'entropy'],
          'min_samples_leaf':[23,24,25],
          'random_state':[123]}

    model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

    acc_score = []


    for train_index , test_index in kf.split(x_train_pca):
      X_train , X_test, y_train , y_test = x_train_pca[train_index],x_train_pca[test_index], Ytr[train_index] , Ytr[test_index]

      model1.fit(X_train,y_train)

      print("Best Hyper Parameters:",model1.best_params_)

      pred_values = model1.predict(X_test)

      acc = accuracy_score(pred_values , y_test)

      acc_score.append(acc)

    avg_acc_score = sum(acc_score)/k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))

def Reg_KNN(x_train_pca, Ytr):
    sizes, training_scores, testing_scores = learning_curve(KNeighborsClassifier(n_neighbors=14, metric='minkowski', algorithm = 'brute', leaf_size=15), x_train_pca, Ytr, cv=5, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))

    # Mean and Standard Deviation of training scores
    mean_training = np.mean(training_scores, axis=1)
    Standard_Deviation_training = np.std(training_scores, axis=1)

    # Mean and Standard Deviation of testing scores
    mean_testing = np.mean(testing_scores, axis=1)
    Standard_Deviation_testing = np.std(testing_scores, axis=1)

    # dotted blue line is for training scores and green line is for cross-validation score
    plt.plot(sizes, mean_training, '--', color="b",  label="Training score")
    plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

    # Drawing plot
    plt.title("LEARNING CURVE FOR KNN Classifier")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def Reg_DT(x_train_pca, Ytr):
    sizes, training_scores, testing_scores = learning_curve(DecisionTreeClassifier(random_state=1234, criterion='gini', min_samples_leaf=25), x_train_pca, Ytr, cv=5, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))

    # Mean and Standard Deviation of training scores
    mean_training = np.mean(training_scores, axis=1)
    Standard_Deviation_training = np.std(training_scores, axis=1)

    # Mean and Standard Deviation of testing scores
    mean_testing = np.mean(testing_scores, axis=1)
    Standard_Deviation_testing = np.std(testing_scores, axis=1)

    # dotted blue line is for training scores and green line is for cross-validation score
    plt.plot(sizes, mean_training, '--', color="b",  label="Training score")
    plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

    # Drawing plot
    plt.title("LEARNING CURVE FOR KNN Classifier")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def Reg(x_train_pca, Ytr):
    Reg_KNN(x_train_pca, Ytr)
    Reg_DT(x_train_pca, Ytr)

def Train_model(x_train_pca, Ytr):
    KNN_model(x_train_pca, Ytr)
    DT_model(x_train_pca, Ytr)

def save_model():
    model = KNeighborsClassifier(n_neighbors=14, metric='minkowski', algorithm = 'brute', leaf_size=15)
    model.fit(x_train_pca,Ytr)

    # Save to file
    pkl_filename = "my_model.pkl"
    with open(pkl_filename, 'wb') as file:
      pickle.dump(model, file)

def main():
    x_train_pca, x_test_pca, Ytr = load_data()
    Train_model(x_train_pca, Ytr)
    Reg(x_train_pca, Ytr)
    save_model()



if __name__ == '__main__':
    main()
