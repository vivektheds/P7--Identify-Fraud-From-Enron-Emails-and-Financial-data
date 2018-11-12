#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','to_messages', 'deferred_income','deferral_payments','restricted_stock_deferred','loan_advances',
       'salary', 'total_stock_value', 'expenses', 'restricted_stock','director_fees',
       'from_poi_to_this_person', 'long_term_incentive',
       'from_this_person_to_poi', 'total_payments', 'exercised_stock_options',
       'other', 'from_messages', 'bonus', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Create new feature(s)

def ComputeMsgFraction(poi_messages, all_messages):
    fraction = 0
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = poi_messages/float(all_messages)

    return fraction

for name in my_dataset:
    from_this_person_to_poi = my_dataset[name]['from_this_person_to_poi']
    from_messages = my_dataset[name]['from_messages']
    fraction_to_poi = ComputeMsgFraction(from_this_person_to_poi, from_messages)
    my_dataset[name]['fraction_to_poi'] = fraction_to_poi
    
    from_poi_to_this_person = my_dataset[name]['from_poi_to_this_person']
    to_messages = my_dataset[name]['to_messages']
    fraction_from_poi = ComputeMsgFraction(from_poi_to_this_person, to_messages)
    my_dataset[name]['fraction_from_poi'] = fraction_from_poi
 
features_list1 = ['poi','to_messages', 'deferred_income','deferral_payments','restricted_stock_deferred','loan_advances',
       'salary', 'total_stock_value', 'expenses', 'restricted_stock','director_fees',
       'from_poi_to_this_person', 'long_term_incentive',
       'from_this_person_to_poi', 'total_payments', 'exercised_stock_options',
       'other', 'from_messages', 'bonus', 'shared_receipt_with_poi', 'fraction_to_poi','fraction_from_poi']   


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list1, sort_keys = True)
labels, features = targetFeatureSplit(data)


# take second element for sort
def takeSecond(elem):
    return elem[1]

# intelligently select features (univariate feature selection)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 5)
selector.fit(features, labels)
scores = zip(features_list1[1:], selector.scores_)
sorted_scores = sorted(scores, key = takeSecond, reverse = True)
#print 'SelectKBest scores: ', sorted_scores

POI_label = ['poi']
kBest_features = POI_label + [(i[0]) for i in sorted_scores[0:5]]
print 'KBest', kBest_features

for name in my_dataset:
    for i in my_dataset[name]:
        if my_dataset[name][i] == 'NaN':
            # fill NaN values
            my_dataset[name][i] = 0



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, cross_validation

data = featureFormat(my_dataset, kBest_features, sort_keys = True)
labels, features= targetFeatureSplit(data)
features = preprocessing.scale(features)
features_train, features_test, labels_train, labels_test = train_test_split( features, labels, test_size = 0.2, random_state = 100)




# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
#clf = tree.DecisionTreeClassifier(criterion='gini',splitter='random',min_samples_split=10,max_depth=None)
clf = tree.DecisionTreeClassifier()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, kBest_features)
