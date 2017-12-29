from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
x, y = iris.data, iris.target
clf.fit(x, y)

# method 1  -> use pickle
import pickle

# save
with open('sk_save/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)

# restore
with open('sk_save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)

    print(clf2.predict(x[:1]))


# method 2  -> use joblib
from sklearn.externals import joblib
# save
joblib.dump(clf, 'sk_save/clf.pkl')

# restore
clf3 = joblib.load('sk_save/clf.pkl')

print(clf3.predict(x[:1]))


