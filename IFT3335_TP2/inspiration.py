# DT decision tree
def DT(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

# Naive Bayes
def NB(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    gnb = MultinomialNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

# forest od random trees
def RF(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

# SVM
def SVM(X_train, X_test, y_train, y_test):
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

# multi-layer perceptron
def MLP(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:", confusion_matrix(y_test, y_pred))
    print("Classification report:", classification_report(y_test, y_pred))

    