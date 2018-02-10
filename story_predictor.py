import numpy as np 
import pandas as pd 
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from tqdm import tqdm

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


le = preprocessing.LabelEncoder()
auth = le.fit_transform(train.author.values)

xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, auth, train_size=0.75, random_state=20)


''' Predicting using tfidvectorizer'''
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(list(xtrain)+list(xvalid))
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)

lr = LogisticRegression(C=1.0)
lr.fit(xtrain_tfv, ytrain)

ans = lr.predict_proba(xvalid_tfv)

print "The log loss using tfidVectorizer for Logistic Regression is" + str(multiclass_logloss(yvalid,ans))


''' Predicting using Countvectorizer'''
cv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

cv.fit(list(xtrain) + list(xvalid))
xtrain_cv = cv.transform(xtrain)
xvalid_cv = cv.transform(xvalid)

lr.fit(xtrain_cv, ytrain)

ans = lr.predict_proba(xvalid_cv)

print "The log loss using CountVectorizer for Logistic Regression is " + str(multiclass_logloss(yvalid,ans))


# ''' Predicting using Naive Bayes'''

nb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
nb.fit(xtrain_tfv, ytrain)
ans = nb.predict_proba(xvalid_tfv)

print "The log loss using tfidVectorizer for Naive Bayes is " + str(multiclass_logloss(yvalid,ans))

nb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
nb.fit(xtrain_cv, ytrain)
ans = nb.predict_proba(xvalid_cv)

print "The log loss using CountVectorizer for Naive Bayes is " + str(multiclass_logloss(yvalid,ans))


# ''' Predicting using svm'''

svc = svm.SVC(C=1.0, kernel='rbf', degree=3, probability=True)
svc.fit(xtrain_tfv, ytrain)
ans = svc.predict_proba(xvalid_tfv)

print "The log loss using tfidVectorizer for SVM is " + str(multiclass_logloss(yvalid,ans))

svc = MultinomialNB(C=1.0, kernel='rbf', degree=3, probability=True)
svc.fit(xtrain_cv, ytrain)
ans = svc.predict_proba(xvalid_cv)

print "The log loss using CountVectorizer for SVM is " + str(multiclass_logloss(yvalid,ans))


''' Predicting using xgboost'''

xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)
xgb.fit(xtrain_tfv, ytrain)
ans = xgb.predict_proba(xvalid_tfv)

print "The log loss using tfidVectorizer for xgboost is " + str(multiclass_logloss(yvalid,ans))

xgb.fit(xtrain_cv, ytrain)
ans = xgb.predict_proba(xvalid_cv)

print "The log loss using CountVectorizer for xgboost is " + str(multiclass_logloss(yvalid,ans))

Grid Search

mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)

svd = TruncatedSVD()
scl = preprocessing.StandardScaler()
lr_model = LogisticRegression()
clf = pipeline.Pipeline([('svd',svd),('scl',scl),('lr',lr_model)])

param_grid = {'svd__n_components' : [120,180], 'lr__C':[0.1,1.0,10], 'lr__penalty':['l1','l2']}
model = GridSearchCV(estimator=clf, param_grid=param_grid,scoring=mll_scorer,verbose=10,n_jobs=-1,iid=True,refit=True,cv=2)

model.fit(xtrain_tfv,ytrain)

print "Best possible score is " + str(model.best_score_)

nb_model = MultinomialNB()

# # Create the pipeline 
clf = pipeline.Pipeline([('nb', nb_model)])

# # parameter grid
param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# # Initialize Grid Search Model
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# # Fit Grid Search Model
model.fit(xtrain_cv, ytrain)  # we can use the full data here but im only using xtrain. 
print("Best score: %0.3f" % model.best_score_)

