# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
#%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

full = train.append(test, ignore_index=True)
# print full['Survived']
titanic = full[:891]
# print titanic, full

# print titanic.head()
#print titanic.describe()
sex = pd.Series(np.where(full.Sex=='male',1,0), name='Sex')
# print sex

embarked = pd.get_dummies(full.Embarked,prefix='Embarked')
# print embarked

pclass = pd.get_dummies(full.Pclass, prefix='Pclass')
# print pclass

imputed = pd.DataFrame()

imputed['Age'] = full.Age.fillna(full.Age.mean())
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())

cabin = pd.DataFrame()

cabin['Cabin'] = full.Cabin.fillna('U')
# # cabin['Cabin'] = cabin['Cabin'].map(lambda c: c[0])
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )
cabin = pd.get_dummies(cabin['Cabin'],prefix='Cabin')
# print cabin.head()

# def cleanTicket(ticket):
#     ticket = ticket.replace('.','')
#     ticket = ticket.replace('/','')
#     ticket = ticket.split()
#     ticket = map( lambda t : t.strip() , ticket )
#     ticket = list(filter(lambda t: not t.isdigit(), ticket))
#     if len(ticket) > 0:
#         return ticket[0]
#     else:
#         return 'XXX'

# ticket = pd.DataFrame()

# ticket['Ticket'] = full['Ticket'].map(cleanTicket)
# ticket['Ticket'] = pd.get_dummies(ticket['Ticket'],prefix='Ticket')

# print ticket.head()

family = pd.DataFrame()

family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

# print family.head()


full_X = pd.concat([imputed, cabin, embarked, sex, pclass], axis=1)
# print full_X.head()

train_valid_X = full_X[:891]
train_valid_Y = titanic['Survived']
test_X = full_X[891:]
test_Y = full['Survived']

train_X, test_X_1, train_Y, test_Y_1 = train_test_split(train_valid_X, train_valid_Y, train_size = 0.85)

# print test_X.shape, test_Y.shape  

logistic_reg = LogisticRegression()
logistic_reg.fit(train_X, train_Y)
print logistic_reg.score(test_X_1, test_Y_1)
# test_Y = logistic_reg.predict(test_X)


# result.to_csv('result.csv', encoding='utf-8', index=False)
Svc = SVC()
Svc.fit(train_X,train_Y)
print Svc.score(test_X_1,test_Y_1)
# test_Y = Svc.predict(test_X)

model = RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_Y)
print model.score(test_X_1,test_Y_1)
# test_Y = model.predict(test_X)

rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_Y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_X , train_Y )
print rfecv.score(test_X_1,test_Y_1)
test_Y = rfecv.predict(test_X)

passenger_id = full[891:].PassengerId
test = pd.DataFrame({'PassengerId':passenger_id, 'Survived': test_Y})
print test.shape
test.to_csv('pred.csv',index = False)