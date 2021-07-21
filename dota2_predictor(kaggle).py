import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import datetime

def gb(x, y):
    for n in [10,20,30,50]:
        print(f'n_estimators={n}')
        start = datetime.datetime.now()
        model = GradientBoostingClassifier(n_estimators=n)
        quality = cross_val_score(model, x, y, cv=kf, scoring='roc_auc').mean()
        print(f'quality={quality}')
        print(datetime.datetime.now()-start)

def lr(x, y):
    for i in range(-5,6):
        c = 10**i
        print(f'C={c}')
        start = datetime.datetime.now()
        model = LogisticRegression(C=c)
        quality = cross_val_score(model, x, y, cv=kf, scoring='roc_auc').mean()
        print(f'quality={quality}')
        print(datetime.datetime.now()-start)

def unique_heroes(df):
    heroes_id = np.unique(df[hero_pick_columns]).tolist()
    unique_heroes_num = len(heroes_id)
    quantity_of_heroes = max(heroes_id)
    return unique_heroes_num, quantity_of_heroes

def heroes_pick(df, n):
    X_pick = np.zeros((df.shape[0], n))

    for i, match_id in enumerate(df.index):
        for p in range(5):
            X_pick[i, df.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, df.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
            
    X_pick = pd.DataFrame(X_pick, index=df.index,
                      columns=[f'hero_{i+1}' for i in range(quantity_of_heroes)])
    return X_pick

features = pd.read_csv('features.csv', index_col='match_id')
features.drop(['duration', 'tower_status_radiant', 'tower_status_dire',
            'barracks_status_radiant', 'barracks_status_dire'],
            axis=1, inplace=True)

missing_features = [len(features)-i for i in features.count()]
with open('missing_features.txt', 'w') as file:
    for index, value in enumerate(missing_features):
        if value != 0:
            file.write(list(features)[index]+', ')

for i in list(features):
    features[i].fillna(int(np.mean(features[i])), inplace=True)

X_train = features.drop('radiant_win', axis=1)
Y_train = features['radiant_win']

kf = KFold(n_splits=5, shuffle=True)
print('GradientBoostingClassifier')
gb(X_train, Y_train)

#for LogisticRegression
features.fillna(0, inplace=True)

X_train_lr = StandardScaler().fit_transform(X_train)
print('LogicticRegression')
lr(X_train_lr, Y_train)

hero_pick_columns = [f'r{i}_hero' for i in range(1,6)]+\
                     [f'd{j}_hero' for j in range(1,6)]
categorical_features = ['lobby_type'] + hero_pick_columns
X_train_wo_cf = X_train.drop(categorical_features, axis=1)
X_train_wo_cf_copy = X_train_wo_cf.copy()
X_train_wo_cf = StandardScaler().fit_transform(X_train_wo_cf)

print('\nlr wo categorical features')
lr(X_train_wo_cf, Y_train)

unique_heroes_num, quantity_of_heroes = unique_heroes(features)

X_pick = heroes_pick(features, quantity_of_heroes)

X_train_w_bag = pd.concat([X_train_wo_cf_copy, X_pick], axis=1)
X_train_w_bag = StandardScaler().fit_transform(X_train_w_bag)

print('\nlr with X_pick')
lr(X_train_w_bag, Y_train)

model = LogisticRegression(C=0.001)
model.fit(X_train_w_bag, Y_train)

test_data = pd.read_csv('features_test.csv', index_col='match_id')
test_data.fillna(0, inplace=True)

unique_heroes_num, quantity_of_heroes = unique_heroes(test_data)

X_pick = heroes_pick(test_data, quantity_of_heroes)
test_data.drop(categorical_features, axis=1, inplace=True)

X_test = pd.concat([test_data, X_pick], axis=1)
X_test = StandardScaler().fit_transform(X_test)

predictions = model.predict_proba(X_test)
max_pred = np.max(predictions)
min_pred = np.min(predictions)
print(f'max_pred = {max_pred}')
print(f'min_pred = {min_pred}')   

