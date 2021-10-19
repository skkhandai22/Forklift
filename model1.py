import pickle
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def model1(filename):
    df = pd.read_csv(filename)

    df['AmountOfWorkDone'] = ''
    for i in range(df.shape[0]):
        df['AmountOfWorkDone'][i] = (df['Total_Work_Order_Completed'][i] / df['Total_Work_order_Received'][i]) * 100

    col1 = 'Service_Required'
    col2 = 'AmountOfWorkDone'
    df = df[[col1 if col == col2 else col2 if col == col1 else col for col in df.columns]]

    df.to_csv("Updated-file.csv", encoding='utf-8')

    objFeatures = df.select_dtypes(include="object").columns
    le = preprocessing.LabelEncoder()

    for feat in objFeatures:
        df[feat] = le.fit_transform(df[feat])

    y = df.iloc[:, -1:]
    x = df.iloc[:, 0:-1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    # print(x, y)
    #sc = StandardScaler()
    #x_train = sc.fit_transform(x_train)
    #x_test = sc.transform(x_test)
    selected_model = []
    selected_model.append(('LR', LogisticRegression()))
    # selected_model.append(('LDA', LinearDiscriminantAnalysis()))
    selected_model.append(('K Nearest  Neighbor', KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=1)))
    selected_model.append(('CART', DecisionTreeClassifier()))
    selected_model.append(('Random Forest', RandomForestClassifier(n_estimators=100, criterion="entropy")))
    selected_model.append(('SVM', SVC()))

    results = []
    names = []
    print(selected_model)
    scoring = 'accuracy'
    for name, model in selected_model:
        kfold = model_selection.KFold(n_splits=10)
        test_results = model_selection.cross_val_score(model, x_train, y_train.values.reshape(-1, ), cv=kfold,
                                                       scoring=scoring)
        results.append(test_results)
        names.append(name)
        print("%s: Mean Accuracy = %.2f%% - SD Accuracy = %.2f%%" % (
            name, test_results.mean() * 100, test_results.std() * 100))

    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)
    print(x_test.shape)
    y_pred = classifier.predict(x_test)
    print(y_pred.shape)
    return y_pred,x_test,df