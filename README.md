
## Identify Fraud From Enron Emails and Financial data
### By Vivek Pandey

### Enron Fraud detection Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

In this project,I will use Machine learning algoritham to identify Person of interests (POIs) which were involved in fraud activity happened in Enron company.

### INTRODUCTION:
#### Background of Enron Dataset:
This dataset contains Enron emails sent and received by Enron executives during 2000-2002 and this dataset has email metadata such as no of emails received and no of emails sent, moreover this dataset contains financial information such as Salary, Bonus and stock options.

As Fortune named Enron "America's Most Innovative Company" for six consecutive years ,however after that Enron was declared bankrupt due to willful corporate fraud and corruption, which alarmed every company to take fraud very seriously and try to take some serious action to prevent such malicious activities within organization.

So the goal of this project to build a predictive model to identify POIs with he help of Machine learning algorithms and using Enron dataset. 
Since this dataset also has actual POIs and using these actual values we will evaluate our predictive model.


The features in the data fall into three major types, namely financial features, email features and POI labels.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)


```python

import pandas as pd
import numpy as np

import sys
import pickle
sys.path.append("C:/Users/Vivek/ud120-projects-master_Final_Vivek/ud120-projects-master/tools/")


from feature_format import featureFormat, targetFeatureSplit
##from tester_plus import dump_classifier_and_data

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

Enron_dictionary = pickle.load( open("C:/Users/Vivek/ud120-projects-master_Final_Vivek/ud120-projects-master/final_project/final_project_dataset.pkl", "r") )

```

### Data exploration :


```python
# Convert Data dictionary to pandas dataframe for easy data manipulation
enron_DF = pd.DataFrame.from_dict(Enron_dictionary, orient = 'index')
```


```python
# Print first few records to check for data and formats
enron_DF.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>bonus</th>
      <th>restricted_stock</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>loan_advances</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>director_fees</th>
      <th>deferred_income</th>
      <th>long_term_incentive</th>
      <th>email_address</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201955</td>
      <td>2902</td>
      <td>2869717</td>
      <td>4484442</td>
      <td>1729541</td>
      <td>4175000</td>
      <td>126027</td>
      <td>1407</td>
      <td>-126027</td>
      <td>1729541</td>
      <td>...</td>
      <td>NaN</td>
      <td>2195</td>
      <td>152</td>
      <td>65</td>
      <td>False</td>
      <td>NaN</td>
      <td>-3081055</td>
      <td>304805</td>
      <td>phillip.allen@enron.com</td>
      <td>47</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>178980</td>
      <td>182466</td>
      <td>257817</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477</td>
      <td>566</td>
      <td>NaN</td>
      <td>916197</td>
      <td>4046157</td>
      <td>NaN</td>
      <td>1757552</td>
      <td>465</td>
      <td>-560222</td>
      <td>5243487</td>
      <td>...</td>
      <td>NaN</td>
      <td>29</td>
      <td>864523</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>-5104</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>39</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102</td>
      <td>NaN</td>
      <td>1295738</td>
      <td>5634343</td>
      <td>6680544</td>
      <td>1200000</td>
      <td>3942714</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10623258</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2660303</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>-1386055</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239671</td>
      <td>NaN</td>
      <td>260455</td>
      <td>827696</td>
      <td>NaN</td>
      <td>400000</td>
      <td>145796</td>
      <td>NaN</td>
      <td>-82782</td>
      <td>63014</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>-201641</td>
      <td>NaN</td>
      <td>frank.bay@enron.com</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# No of Rows and Columns
print enron_DF.shape

```

    (146, 21)
    

Dataset has data for 146 people and there are 20 features,since "poi" column  is our label which tells us whether person is POI hence you are excluding this coulmn in our prediction model.


```python

print "There are total {} people in the dataset." .format(enron_DF.shape[0]) 
print "Out of them there are {} POIs and {} Non POIs." .format(enron_DF[enron_DF.poi==True].shape[0] , enron_DF[enron_DF.poi==False].shape[0]) 

```

    There are total 146 people in the dataset.
    Out of them there are 18 POIs and 128 Non POIs.
    


```python
#quick overview of the data in the dataset
enron_DF.info()
enron_DF.describe().transpose()

```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
    Data columns (total 21 columns):
    salary                       146 non-null object
    to_messages                  146 non-null object
    deferral_payments            146 non-null object
    total_payments               146 non-null object
    exercised_stock_options      146 non-null object
    bonus                        146 non-null object
    restricted_stock             146 non-null object
    shared_receipt_with_poi      146 non-null object
    restricted_stock_deferred    146 non-null object
    total_stock_value            146 non-null object
    expenses                     146 non-null object
    loan_advances                146 non-null object
    from_messages                146 non-null object
    other                        146 non-null object
    from_this_person_to_poi      146 non-null object
    poi                          146 non-null bool
    director_fees                146 non-null object
    deferred_income              146 non-null object
    long_term_incentive          146 non-null object
    email_address                146 non-null object
    from_poi_to_this_person      146 non-null object
    dtypes: bool(1), object(20)
    memory usage: 24.1+ KB
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>salary</th>
      <td>146</td>
      <td>95</td>
      <td>NaN</td>
      <td>51</td>
    </tr>
    <tr>
      <th>to_messages</th>
      <td>146</td>
      <td>87</td>
      <td>NaN</td>
      <td>60</td>
    </tr>
    <tr>
      <th>deferral_payments</th>
      <td>146</td>
      <td>40</td>
      <td>NaN</td>
      <td>107</td>
    </tr>
    <tr>
      <th>total_payments</th>
      <td>146</td>
      <td>126</td>
      <td>NaN</td>
      <td>21</td>
    </tr>
    <tr>
      <th>exercised_stock_options</th>
      <td>146</td>
      <td>102</td>
      <td>NaN</td>
      <td>44</td>
    </tr>
    <tr>
      <th>bonus</th>
      <td>146</td>
      <td>42</td>
      <td>NaN</td>
      <td>64</td>
    </tr>
    <tr>
      <th>restricted_stock</th>
      <td>146</td>
      <td>98</td>
      <td>NaN</td>
      <td>36</td>
    </tr>
    <tr>
      <th>shared_receipt_with_poi</th>
      <td>146</td>
      <td>84</td>
      <td>NaN</td>
      <td>60</td>
    </tr>
    <tr>
      <th>restricted_stock_deferred</th>
      <td>146</td>
      <td>19</td>
      <td>NaN</td>
      <td>128</td>
    </tr>
    <tr>
      <th>total_stock_value</th>
      <td>146</td>
      <td>125</td>
      <td>NaN</td>
      <td>20</td>
    </tr>
    <tr>
      <th>expenses</th>
      <td>146</td>
      <td>95</td>
      <td>NaN</td>
      <td>51</td>
    </tr>
    <tr>
      <th>loan_advances</th>
      <td>146</td>
      <td>5</td>
      <td>NaN</td>
      <td>142</td>
    </tr>
    <tr>
      <th>from_messages</th>
      <td>146</td>
      <td>65</td>
      <td>NaN</td>
      <td>60</td>
    </tr>
    <tr>
      <th>other</th>
      <td>146</td>
      <td>93</td>
      <td>NaN</td>
      <td>53</td>
    </tr>
    <tr>
      <th>from_this_person_to_poi</th>
      <td>146</td>
      <td>42</td>
      <td>NaN</td>
      <td>60</td>
    </tr>
    <tr>
      <th>poi</th>
      <td>146</td>
      <td>2</td>
      <td>False</td>
      <td>128</td>
    </tr>
    <tr>
      <th>director_fees</th>
      <td>146</td>
      <td>18</td>
      <td>NaN</td>
      <td>129</td>
    </tr>
    <tr>
      <th>deferred_income</th>
      <td>146</td>
      <td>45</td>
      <td>NaN</td>
      <td>97</td>
    </tr>
    <tr>
      <th>long_term_incentive</th>
      <td>146</td>
      <td>53</td>
      <td>NaN</td>
      <td>80</td>
    </tr>
    <tr>
      <th>email_address</th>
      <td>146</td>
      <td>112</td>
      <td>NaN</td>
      <td>35</td>
    </tr>
    <tr>
      <th>from_poi_to_this_person</th>
      <td>146</td>
      <td>58</td>
      <td>NaN</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>



In the above summary table of missing values frequency, there are many features which has more than 50 % NaN values, so I will try not to consider such features which has many missing values.

### Outlier investigation :

In the process of outlier investigation I have identified below three outliers which needs to remove before futher analysis.

1- TOTAL: This is a summary of all enron's executives financial data.

2- THE TRAVEL AGENCY IN THE PARK : This is a company which was co-owned by the sister of Enron's former Chairman, and here we are investigating Enron's executive, hence we should not include it in our dataset.

#### Needs to implement 3- Invalid data points: “LOCKHART EUGENE E” (contains over 95% invalid values). To keep the quality of analysis, these outliers were removed before further analysis.


```python
# Import plotly 
import plotly
from plotly import tools
plotly.tools.set_credentials_file(username='vivektheds', api_key='YobnbIPE6Lwvzb6emY51')

```


```python
import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Scatter(
    x=enron_DF.salary,
    y=enron_DF.bonus,
    mode='markers',
    text=enron_DF.index  
)


data = [trace1]

layout = go.Layout(
    
    xaxis=dict(
        title='Salary',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Bonus',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
    
    #showlegend=True
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='Enron-Salary Vs Bonus')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~vivektheds/2.embed" height="525px" width="100%"></iframe>




```python
enron_DF.loc[['TOTAL']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>bonus</th>
      <th>restricted_stock</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>loan_advances</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>director_fees</th>
      <th>deferred_income</th>
      <th>long_term_incentive</th>
      <th>email_address</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOTAL</th>
      <td>26704229</td>
      <td>NaN</td>
      <td>32083396</td>
      <td>309886585</td>
      <td>311764000</td>
      <td>97343619</td>
      <td>130322299</td>
      <td>NaN</td>
      <td>-7576788</td>
      <td>434509511</td>
      <td>...</td>
      <td>83925000</td>
      <td>NaN</td>
      <td>42667589</td>
      <td>NaN</td>
      <td>False</td>
      <td>1398517</td>
      <td>-27992891</td>
      <td>48521928</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
enron_DF.loc[['THE TRAVEL AGENCY IN THE PARK']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>exercised_stock_options</th>
      <th>bonus</th>
      <th>restricted_stock</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock_deferred</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>loan_advances</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>director_fees</th>
      <th>deferred_income</th>
      <th>long_term_incentive</th>
      <th>email_address</th>
      <th>from_poi_to_this_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>THE TRAVEL AGENCY IN THE PARK</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>362096</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>362096</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



As per the above graph (before outlier removal) between salary and bonus it is clear that there is one label named as 'TOTAL' which seem an outlier as this belongs to summary of all enron's executives financial data.


```python
# Removing outlier
enron_DF.drop(['TOTAL'], inplace= True)
enron_DF.drop(['THE TRAVEL AGENCY IN THE PARK'], inplace= True)
```


```python
# After outlier removal 

trace1 = go.Scatter(
    x=enron_DF.salary,
    y=enron_DF.bonus,
    mode='markers',
    text=enron_DF.index  
)


data = [trace1]

layout = go.Layout(
    
    xaxis=dict(
        title='Salary',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Bonus',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
    
    #showlegend=True
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='Enron-Salary Vs Bonus')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~vivektheds/2.embed" height="525px" width="100%"></iframe>



Even after outlier removal it seems there are still some outliers however those are top executives like kenneth lay, Jef Skilling and Lavorato John (who received cash bonuses of $8 million to keep them from leaving Enron last fall) and they can have big bonus as their salary is also high.


```python
#import plotly.plotly as py
#import plotly.graph_objs as go

trace1 = go.Scatter(
    x=enron_DF.salary,
    y=enron_DF.total_stock_value,
    mode='markers',
    text=enron_DF.index  
)


data = [trace1]

layout = go.Layout(
    
    xaxis=dict(
        title='Salary',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='total stock value',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
    
    #showlegend=True
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='Enron-Salary Vs total stock value')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~vivektheds/4.embed" height="525px" width="100%"></iframe>



From the above Salary and Total stock value graph it also stand out few of other names as well like Rice Kenneth D and White JR thomas K who got 22 M and 15 M stock values however these figures are susceptible with respect to their salary.

## Feature Engineering :

In feature engineering in addition to 21 features I have created three new features from existing features that could improve the performance of Enron POI prediction.

- from_poi_ratio: fraction of emails from POIs,which represents the ratio of the messages from POI to this person against all the messages sent to this person

- to_poi_ratio: fraction of emails to POIs,which represents ratio from this person to POI against all messages from this person.

- shared_poi_ratio: the ratio of email receipts shared with a person of interest to all emails addressed to that individual. 


The rational behind these choices is that the absolute no of emails from/to a POIs might be misleading figure because there might be some innocent employees who just sent/received many emails from POIs as part of your daily official work, however
fraction of these emails to total message may help us to identify the TRUE POIs since corporate fraud can not be done alone and it requires a set of people to do that at larger scale.


```python
# Add the new email features to the dataframe
enron_DF.replace(to_replace='NaN', value=0.0, inplace=True)
enron_DF['to_poi_ratio'] = enron_DF['from_poi_to_this_person']/enron_DF['to_messages']
enron_DF['from_poi_ratio'] = enron_DF['from_this_person_to_poi'] / enron_DF['from_messages']
enron_DF['shared_poi_ratio'] = enron_DF['shared_receipt_with_poi'] / enron_DF['to_messages']

enron_DF.replace(to_replace='NaN', value=0.0, inplace=True)
```

## Feature Scaling : 

It is used to standardize the data of all features. As all features had different units and some of the features had very high value which can have undue influence on the classifier hence in order to make data standard/scaled for classifiers I have used preprocessing.scale() function from sklearn to scale all features to a range between 0 and 1.



```python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import classification_report

x1 = enron_DF.drop(['email_address','poi'], axis=1)
x1 = preprocessing.scale(x1)
y1=enron_DF['poi']
```

## Feature Selection (intelligently select features)

Feature selection is very important aspect because all features does not equally participate in the prediction and that can add noisy data to our prediction model, so it's very important to feed right set of features which can give us better prediction results.

For feature selection first I have used an extremely randomized tree classifier and in this classifier there is a featureimportances attributes that is the feature importances (the higher, the more important the feature), so using this approach I am trying to find out the no of features which I shoud use for better classification results.


```python
# Plot number of features VS. cross-validation scores
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# feature extraction
model = ExtraTreesClassifier()
model.fit(x1, y1)
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x1.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

#plt.figure(1, figsize=(14, 13))
#plt.title("Feature importances")
#plt.bar(range(x1.shape[1]), importances[indices],
#       color="g", yerr=std[indices], align="center")
#plt.xticks(range(x1.shape[1]), x1.columns[indices],rotation=90)
#plt.xlim([-1, x1.shape[1]])
#plt.show()
```

    Feature ranking:
    1. feature 20 (0.118967)
    2. feature 4 (0.104833)
    3. feature 5 (0.097338)
    4. feature 9 (0.074302)
    5. feature 16 (0.071997)
    6. feature 6 (0.069789)
    7. feature 0 (0.060055)
    8. feature 10 (0.047674)
    9. feature 13 (0.045733)
    10. feature 7 (0.045384)
    11. feature 1 (0.043485)
    12. feature 3 (0.038973)
    13. feature 14 (0.036888)
    14. feature 21 (0.035770)
    15. feature 18 (0.026772)
    16. feature 12 (0.023747)
    17. feature 2 (0.018037)
    18. feature 19 (0.016129)
    19. feature 17 (0.014256)
    20. feature 11 (0.009400)
    21. feature 15 (0.000316)
    22. feature 8 (0.000154)
    

As per the above feature ranking, after 5 best features importance of features decrease. Therefore I can focus on 5 features.

In order to decide the best features to use, I utilized an automated feature selection function, i.e. SelectKBest, which selects the K features that are defined by the amount of variance explained, automatically selected for use in the classifier. 
Moreover, here I have selected K=5 which i got from the featureimportances attribute.


```python

# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

x1 = enron_DF.drop(['email_address','poi'], axis=1)
#x1 = preprocessing.scale(x1)
y1=enron_DF['poi']

# feature extraction
test = SelectKBest(score_func=f_classif, k=5)
fit = test.fit(x1, y1)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
scores1 = -np.log10(test.pvalues_)
print scores1

features = fit.transform(x1)
# summarize selected features
print(features[0:5,:])

print features.shape
print(test.get_support(indices=True))

#2nd way
x1.columns[test.get_support(indices=True)].tolist()
```

    [ 18.576   1.699   0.217   8.874  25.098  21.06    9.347   8.746   0.065
      24.468   6.234   7.243   0.164   4.246   2.427   2.108  11.596  10.072
       5.345   3.211  16.642   9.296]
    [ 4.518  0.711  0.192  2.468  5.797  5.013  2.573  2.44   0.097  5.677
      1.864  2.098  0.164  1.385  0.915  0.827  3.066  2.734  1.653  1.123
      4.125  2.562]
    [[  2.020e+05   1.730e+06   4.175e+06   1.730e+06   2.961e-02]
     [  0.000e+00   2.578e+05   0.000e+00   2.578e+05   0.000e+00]
     [  4.770e+02   4.046e+06   0.000e+00   5.243e+06   0.000e+00]
     [  2.671e+05   6.681e+06   1.200e+06   1.062e+07   0.000e+00]
     [  2.397e+05   0.000e+00   4.000e+05   6.301e+04   0.000e+00]]
    (144L, 5L)
    [ 0  4  5  9 20]
    




    ['salary',
     'exercised_stock_options',
     'bonus',
     'total_stock_value',
     'from_poi_ratio']



### Pick an algorithm :

To start with I have selected the five algorithms which are Gaussian Naive Bayes (GaussianNB), DecisionTreeClassifier, Support Vector Classifier (SVC),RandomForestClassifier and KNeighborsClassifier,also I have used top five features'salary',
 'exercised_stock_options','bonus', 'total_stock_value','from_poi_ratio' to test for all these five Machine learning algorithams.
 
I will run the algorithms with the default parameters. 

#### DecisionTreeClassifier: 


```python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import classification_report

X = enron_DF.iloc[:, [0,4,5,9,20]]
X = preprocessing.scale(X)

Y = enron_DF.iloc[:,15]
#enron_DF.columns.get_loc("poi")
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

clf_DT = DecisionTreeClassifier()
clf_DT.fit(X_train, y_train)
y_pred = clf_DT.predict(X_test)

print("Report for DecisionTreeClassifier  : ",
    classification_report(y_test,y_pred))

print "Accuracy is ", accuracy_score(y_test,y_pred)*100
```

    ('Report for DecisionTreeClassifier  : ', '             precision    recall  f1-score   support\n\n      False       0.88      0.88      0.88        25\n       True       0.25      0.25      0.25         4\n\navg / total       0.79      0.79      0.79        29\n')
    Accuracy is  79.3103448276
    

#### Gaussian Naive Bayes (GaussianNB):


```python
# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
 
print("Report for Naive Bayes  : ",
    classification_report(y_test,gnb_predictions))

print "Accuracy of Naive Bayes is ", accuracy_score(y_test,gnb_predictions)*100   
```

    ('Report for Naive Bayes  : ', '             precision    recall  f1-score   support\n\n      False       0.89      0.96      0.92        25\n       True       0.50      0.25      0.33         4\n\navg / total       0.84      0.86      0.84        29\n')
    Accuracy of Naive Bayes is  86.2068965517
    

####  Support Vector Classifier (SVC):


```python
# training a linear SVM classifier
from sklearn.svm import SVC # "Support Vector Classifier"
clf_svm_linear = SVC(kernel = 'linear')
clf_svm_linear.fit(X_train, y_train)
svm_predictions = clf_svm_linear.predict(X_test)
 
print("Report for SVM  : ",
    classification_report(y_test,svm_predictions))
print "Accuracy of SVM is ", accuracy_score(y_test,svm_predictions)*100
```

    ('Report for SVM  : ', '             precision    recall  f1-score   support\n\n      False       0.89      1.00      0.94        25\n       True       1.00      0.25      0.40         4\n\navg / total       0.91      0.90      0.87        29\n')
    Accuracy of SVM is  89.6551724138
    

#### RandomForestClassifier:


```python
from sklearn.ensemble import RandomForestClassifier

#rfc = RandomForestClassifier(max_depth=2, random_state=0)
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)

# Use the forest's predict method on the test data
rfc_predictions = rfc.predict(X_test)
print("Report for RandomForestRegressor  : ",
    classification_report(y_test,rfc_predictions))

print "Accuracy of RandomForestRegressor is ", accuracy_score(y_test,rfc_predictions)*100    
```

    ('Report for RandomForestRegressor  : ', '             precision    recall  f1-score   support\n\n      False       0.86      0.96      0.91        25\n       True       0.00      0.00      0.00         4\n\navg / total       0.74      0.83      0.78        29\n')
    Accuracy of RandomForestRegressor is  82.7586206897
    

#### KNeighborsClassifier :


```python
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)
 
knn_predictions = knn.predict(X_test)
print("Report for KNN classifier  : ",
    classification_report(y_test,knn_predictions))

print "Accuracy of KNN classifier is ", accuracy_score(y_test,knn_predictions)*100    
```

    ('Report for KNN classifier  : ', '             precision    recall  f1-score   support\n\n      False       0.86      1.00      0.93        25\n       True       0.00      0.00      0.00         4\n\navg / total       0.74      0.86      0.80        29\n')
    Accuracy of KNN classifier is  86.2068965517
    

#### The results from running the five classifiers with no parameter tuning are summarized below:

##### Classifiers' Summary with default paramters:
|Classifier|precision|recall|f1-score|Accuracy|
|--|-------|---------|------|--------|--------|
|DecisionTreeClassifier |0.79|0.79|0.79|0.79310|
|GaussianNB |0.84|0.86|0.84|0.86206|
|Support Vector Classifier |0.91|0.90|0.87|0.89655|
|RandomForestClassifier |0.74|0.83|0.78|0.82758|
|KNeighborsClassifier |0.74|0.86|0.80|0.86206|

After performing prediction using default parameters on these algorithms I can see that Support Vector Classifier performed best ,followed by the gaussianNB,KNeighborsClassifier,DecisionTreeClassifier and RandomForestClassifier. 

However, there is a lot of scope of improvement on algorithms's parameter tunning.

## Tune the algorithm :

Tunning Machine learning algorithm is the process to select best possible parameters which can result maximum performance on a given dataset for that algorithm.Since Sklearn defines default parameters for each alorithms to give good classifier for as many datasets as possible, hence for different datasets there is a scope of optimizing each algorithm using different parameters.

So we can either manually try different parameters to find out the best parameters for a algorithm or we can use automate it using GridSearchCV which takes all the possible combination of parameters and leverges the best parameters to achieve maximum performance.

#### Find the optimized parameters for SVC:

Since using default Sklearn parameters for each machine learning algorithms Support Vector Classifier had maximum performance in terms of identifing POI, so I will take this algorithm and try optimizing it using GridSearchCV and see if performance gets better.Hence I have created a parameter grind and used different kernels 'linear', 'rbf', 'sigmoid' and other parameter valus for gamma and 'C'.


```python

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import svm
svm_clf = svm.SVC()
svm_param = {'kernel':('linear', 'rbf', 'sigmoid'),
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator = svm_clf, param_grid = svm_param)

# instantiate the grid# instan 
grid_SVM = GridSearchCV(svm_clf, svm_param, cv=10, scoring='accuracy', return_train_score=False)

# fit the grid with data
grid_SVM.fit(X_train,y_train)


# print the array of mean scores only# print  
grid_svm_mean_scores = grid_SVM.cv_results_['mean_test_score']
#print(grid_mean_scores)
# examine the best model# examin 
print(grid_SVM.best_score_)
print(grid_SVM.best_params_)
```

    0.886956521739
    {'kernel': 'rbf', 'C': 1000, 'gamma': 0.001}
    

#### Testing Support Vector Classifier with the above suggested optimized parameters.


```python
 
# training a linear SVM classifier
from sklearn.svm import SVC # "Support Vector Classifier"
clf_svm_linear = SVC(kernel = 'rbf', C = 1000,gamma=0.001)
clf_svm_linear.fit(X_train, y_train)
svm_predictions = clf_svm_linear.predict(X_test)
 
# model accuracy for X_test  
#accuracy_svm = clf_svm_linear.score(y_test, svm_predictions)
#print accuracy_svm
print "Accuracy of SVM using optimized parameters ", accuracy_score(y_test,svm_predictions)*100
print("Report : ",
    classification_report(y_test,svm_predictions))
```

    Accuracy of SVM using optimized parameters  86.2068965517
    ('Report : ', '             precision    recall  f1-score   support\n\n      False       0.86      1.00      0.93        25\n       True       0.00      0.00      0.00         4\n\navg / total       0.74      0.86      0.80        29\n')
    

Using the best suggested model by GridSearchCV we are not getting better results rather it's giving very poor rsult for true prediction cases. So I will be using default linear parameter for my final prediction.

#### Find the optimized parameters for DecisionTreeClassifier:

This classifier builds a model in the form of a tree structure and the biggest challenge is to identify features to split upon and a node to split upon.
So criterion'gini' and 'entropy' are two important parameters which maximize information gain that helps to decide which variable to use when splitting.


```python
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier()
dt_param = {'criterion':('gini', 'entropy'),
'splitter':('best','random'), 'min_samples_split':(2, 4, 6, 8, 10, 20), 'max_depth':(None, 5, 10, 15, 20)}
dt_grid_search = GridSearchCV(estimator = dt_clf, param_grid = dt_param)

# fit the grid with data
dt_grid_search.fit(X_train,y_train)


# print the array of mean scores only# print  
grid_dt_mean_scores = dt_grid_search.cv_results_['mean_test_score']
#print(grid_mean_scores)

# examine the best model# examin 
print(dt_grid_search.best_score_)
print(dt_grid_search.best_params_)


```

    0.895652173913
    {'min_samples_split': 10, 'splitter': 'random', 'criterion': 'gini', 'max_depth': None}
    

#### Testing DecisionTreeClassifier with the above suggested optimized parameters


```python
from sklearn import tree
dt_clf1 = tree.DecisionTreeClassifier(criterion='gini',splitter='random',min_samples_split=10,max_depth=None)
dt_clf1.fit(X_train, y_train)
dt_predict=dt_clf1.predict(X_test)

print "Accuracy of DecisionTreeClassifier using optimized parameters ", accuracy_score(y_test,dt_predict)*100
print("Report : ",
    classification_report(y_test,dt_predict))
```

    Accuracy of DecisionTreeClassifier using optimized parameters  89.6551724138
    ('Report : ', '             precision    recall  f1-score   support\n\n      False       0.89      1.00      0.94        25\n       True       1.00      0.25      0.40         4\n\navg / total       0.91      0.90      0.87        29\n')
    

Using GridSearchCV optimized parameters DecisionTreeClassifier is giving far better result and now True prediction cases as well are predicting with better accuracy.

#### Find the optimized parameters for KNeighborsClassifier:


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
%matplotlib inline


# define the parameter values that should be searched
k_range = list(range(1, 31))

knn = KNeighborsClassifier(n_neighbors=5)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)

# instantiate the grid# instan 
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

# fit the grid with data
grid.fit(X_train,y_train)

# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
```

    0.886956521739
    {'n_neighbors': 9}
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=9, p=2,
               weights='uniform')
    


```python
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=9, p=2,
           weights='uniform').fit(X_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(X_test, y_test)

knn_predictions1 = knn.predict(X_test) 

print "Accuracy of KNN classifier using optimized parameters ", accuracy_score(y_test,knn_predictions1)*100
 
print("Report : ",
    classification_report(y_test,knn_predictions1))
```

    Accuracy of KNN classifier using optimized parameters  86.2068965517
    ('Report : ', '             precision    recall  f1-score   support\n\n      False       0.86      1.00      0.93        25\n       True       0.00      0.00      0.00         4\n\navg / total       0.74      0.86      0.80        29\n')
    

It seems using GridSearchCV optimized parameters KNN classified is giving same result as default parameters and there is no significance difference.

##### Note: Using automated GridSearchCV algorithm I got the maximum performance in DecisionTreeClassifier and got best results for True prediction cases as well, moreover I have used below optimized parameters for DecisionTreeClassifier.

{'min_samples_split': 10, 'splitter': 'random', 'criterion': 'gini', 'max_depth': None}

## Validation Strategy :

Validation is a process to evaluate a trained prediction model on a testing dataset.The classic mistake in validation is over-fitting where our model performs better on training data however model performs badly on testing data (unseen data).
In order to overcome over-fitting problem I have used train_test_split function of sklearn cross_validation algorithm to split training and testing data with 100 trials into 80% of training data and 20% of testing data to train and test model in different sample dataset.

## Evaluation Metrics

In order to evaluate the performance of classifier model we use accuracy which can be a raw measure and it can suit for some dataset however it may not suit for other dataset such as Enron, and the reason is if a model classifies Non-POI so well in Enron dataset and not classifies POIs accurately then still accuracy of the model will be high which can be misleading since actual purpose of making classifier model will not be served.

Hence in order to evaluate such classifiers we should use precision, recall and f1 score as metrics.

Precision = True Positive / (True Positive + False Positive)

Precision is the no of true POI identified divided by total no of POI identified by a algorithm.

Recall = True Positive / (True Positive + False Negative)

Recall is the no of true POI identified by a algorithm divided by total no of POI in the dataset.

F1 Score= 2x(Precision x Recall) / (Precision + Recall)

F1 score is the harmonic average of the precision and recall where F1 score reaches it's best at 1(perfect precision & recall) and worst at 0.

The final results for my classifier model are summarized below:

Both DecisionTreeClassifier with the tunned paramters and Support Vector Classifier with default parameters are giving same best result out of those five algorithms.

##### Final Classifiers' Summary:
|Classifier|precision|recall|f1-score|Accuracy|
|--|-------|---------|------|--------|--------|
|DecisionTreeClassifier with tunned parameters |0.91|0.90|0.87|0.89655|
|Support Vector Classifier with default parameters |0.91|0.90|0.87|0.89655|


## References:

https://www.nytimes.com/2002/06/18/business/officials-got-a-windfall-before-enron-s-collapse.html

https://www.risk.net/risk-management/2123422/ten-years-after-its-collapse-enron-lives-energy-markets

https://en.wikipedia.org/wiki/Enron

https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier


```python

```
