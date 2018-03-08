## 决策树模型完成分类问题

import pandas as pd
from sklearn import preprocessing
from sklearn import tree

adult_data = pd.read_csv('./DecisionTree.csv')

# print(adult_data.head(5))
# print(adult_data.info())
# print(adult_data.shape)

# print(adult_data.columns)
# Index(['workclass', 'education', 'marital-status', 'occupation',
#        'relationship', 'race', 'gender', 'native-country', 'income'],
#       dtype='object')

## 区分一下特征和目标

feature_columns = ['workclass', 'education', 'marital-status',
                   'occupation', 'relationship', 'race',
                   'gender', 'native-country']
label_column = ['income']

features = adult_data[feature_columns]
label = adult_data[label_column]

# print(features.head(2))
#            workclass   education       marital-status        occupation  \
# 0          State-gov   Bachelors        Never-married      Adm-clerical
# 1   Self-emp-not-inc   Bachelors   Married-civ-spouse   Exec-managerial
#
#      relationship    race gender  native-country
# 0   Not-in-family   White   Male   United-States
# 1         Husband   White   Male   United-States

# print(label.head(2))
#    income
# 0   <=50K
# 1   <=50K

# 特征工程

features = pd.get_dummies(features)
# print(features.head(2))

# 构建模型

# 初始化一个决策树分类器
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
# 使用决策树分类器拟合数据
clf = clf.fit(features.values, label.values)

result = clf.predict(features.values)

import pydotplus
from IPython.display import display, Image

dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names=features.columns,
                                class_names = ['<=50k', '>50k'],
                                filled = True,
                                rounded =True
                               )

graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))


