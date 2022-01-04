import pandas as pd
from collections import Counter
import math
#import pprint
#import json
import yaml   #pip install pyyaml

df_tennis=pd.read_csv("id3V2/Company.csv")

def entropy_list(a_list):
    cnt=Counter(x for x in a_list)
    num_instance=len(a_list)
    probs=[x/num_instance for x in cnt.values()]
    return entropy(probs)

def entropy(probs):
    return sum([-prob*math.log(prob,2) for prob in probs])

def info_gain(df,attr,target):
    df_split=df.groupby(attr)   #splitting the dataset into two parts, in which one dataset consist of a column values of attribute 'attr'
    print(df_split.head())
    return
    '''nobs=len(df.index)     #no. of rows in the dataset
    df_agg_ent=df_split.agg({target:[entropy_list, lambda x: len(x)/nobs]})     
    #print(df_agg_ent)
    df_agg_ent.columns=['Entropy','PropObserved']   #adding columns names to the dataset 'df_agg_ent'
    new_entropy=sum(df_agg_ent['Entropy']*df_agg_ent["PropObserved"])
    old_entropy=entropy_list(df[target])
    return old_entropy-new_entropy'''

def id3(df,attribute_name,target):
    cnt=Counter(x for x in df[target])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or (not attribute_name):
        return None
    else:
        #default_class=max(cnt.keys())
        #print(attribute_name)
        gains=[info_gain(df,attr,target) for attr in attribute_name]  #gains stores ig values for each atttributes in dataset
        index_max=gains.index(max(gains))
        best_attr=attribute_name[index_max]
        tree={best_attr:{}}   #creating empty dict
        remaining_attr=[x for x in attribute_name if x!=best_attr]
        #remaining_attr=attribute_name.remove(best_attr)
        for attr_val,data_subset in df.groupby(best_attr):
            subtree=id3(data_subset,remaining_attr,target)
            tree[best_attr][attr_val]=subtree
        return tree

'''def classify(instance,tree,default = None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result,dict):
            return classify(instance,result)
        else:
            return result
    else:
        return default'''

attribute_names=list(df_tennis.columns.str.strip())
attribute_names.remove('Profit')
info_gain(df_tennis,'Competition','Profit')
'''tree=id3(df_tennis,attribute_names,'Profit')
print("The resulant Decision Tree")'''
#pprint.pprint(tree)
#print(json.dumps(tree,sort_keys=False,indent=2))
print(yaml.dump(tree,sort_keys=False,default_flow_style=False))

"""
training_data = df_tennis.iloc[1:-4] # all but last thousand instances
test_data = df_tennis.iloc[-4:] # just the last thousand
train_tree = id3(training_data,  attribute_names,'Profit')
print("\n\nThe Resultant Decision train_tree is :\n")
print(train_tree)
test_data['predicted2'] = test_data.apply(classify,axis=1,args=(train_tree,'Yes') )
print ('\n\n Training the model for a few samples, and again predicting \'Profit\' for remaining attribute')
print('The Accuracy for new trained data is : ' + str( sum(test_data['Profit']==test_data['predicted2'] ) / (1.0*len(test_data.index)) ))"""