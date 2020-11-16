import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
from IPython.display import display

data = pd.read_csv('../indian_food.csv', header=0)
print('Dataset size is %s' % str(data.shape))
print('Dataset coloumns are %s' % list(data.columns))
# display(data)
# print(data['diet'].unique())

data[(data == -1) | (data == '-1')] = np.nan
print('Columns %s has missing (NA) values.')
list(data.columns[data.isna().any()])
# display(data)
print('Mean of prep and cook time: \n' + str(data.mean().round()))
data_2 = data.fillna(data.mean().round())
data_2['ingredients'] = data_2['ingredients'].apply(lambda x: x.lower().replace(', ', ','))
data_3 = data_2.copy()
data_3['ingredients'] = data_3['ingredients'].apply(lambda x: x.split(','))
# display(data_2)


def filter_func(ingredients=None, diet=None, course=None, max_prep_time=None, max_cook_time=None):
    """
    Filter the indian food dataset based on 4 criteria: ingredients, diet, course, max_prep_time, max_cook_time.
    At fist all input are checked to be of the right format and within the acceptable values. This process is not
    100% error proof but can prevent elementary input errors. Then, if a valid input is passed for each criteria, the
    dish dataframe is filter based on. The 'ingredients' criteria is can be passed as a string or list of strings.
    It can be a single string or list/tuple of strings. The dataframe will be filtered for all ingredients in the
    passed ingredients list. Below examples are all correct application of function.
    filter_func(ingredients='oil', max_prep_time=100, max_cook_time=100, diet='non vegetarian')
    filter_func(ingredients=['oil', 'rice'], max_prep_time=100, max_cook_time=100, diet='vegetarian', course='snack')
    filter_func(ingredients='brown rice')

    :param ingredients: The string of one ingredient or list/tuple of strings of multiple ingredients.
    :param diet: The diet to be 'vegetarian' or 'non vegetarian'
    :param course: The course for which the dish should be served can be 'dessert', 'main course', 'starter', 'snack'
    :param max_prep_time: The maximum preparation rime
    :param max_cook_time: The maximum cooking time
    """
    global data_2
    data_term = data_2.copy()
    data_term['ingredients'] = data_term['ingredients'].apply(lambda x: [w.lower().strip() for w in x.split(',')])

    # Filtering based on diet
    if diet is not None:
        if diet.lower() not in data_2['diet'].unique():
            print('Diet input "%s" not valid. You may choose from: %s' % (diet, data_2['diet'].unique()))
            return
        else:
            print('Filtering dishes based on diet to be %s.' % diet)
            data_term = data_term[data_term['diet'] == diet]

    # Filtering based on course
    if course is not None:
        if course.lower() not in data_2['course'].unique():
            print('Course input "%s" not valid. You may choose from: %s' % (course, data_2['course'].unique()))
            return
        else:
            print('Filtering dishes based on course type to be %s.' % course)
            data_term = data_term[data_term['course'] == course]

    # Filtering based on max prep time
    if max_prep_time is not None:
        try:
            max_prep_time = int(max_prep_time)
            print('Filtering dishes with max prep time of %d.' % max_prep_time)
            data_term = data_term[(data_term['prep_time'] <= max_prep_time)]
        except ValueError:
            print('Incorrect input for prep time (%s)' % max_prep_time)
            return

    # Filtering based on maximum cooking time
    if max_cook_time is not None:
        try:
            max_cook_time = int(max_cook_time)
            print('Filtering dishes with max cook time of %d.' % max_cook_time)
            data_term = data_term[(data_term['cook_time'] <= max_cook_time)]
        except ValueError:
            print('Incorrect input for cook time (%s)' % max_cook_time)
            return
    print('_'*80)

    if ingredients is None:
        print('There are %d dishes which met above criteria/s.' % len(data_term))
        display(data_2.loc[data_term.index].head())
        return
    if type(ingredients) is str:
        if ingredients.strip() == '':
            print('There are %d dishes which met above criteria/s.' % len(data_term))
            display(data_2.loc[data_term.index].head())
            return
        ingred_list = [ingredients.lower().strip()]
    else:
        ingred_list = [x.lower().strip() for x in ingredients]

    data_word = data_term.copy()
    data_word['ingredients'] = data_word['ingredients'].apply(lambda x: set(sum([s.split(' ') for s in x], [])))

    for ing in ingred_list:
        if len(data_term) > 0:
            data_term = data_term[data_term['ingredients'].apply(lambda r: ing in r)]
        if len(data_word) > 0:
            data_word = data_word[data_word['ingredients'].apply(lambda r: ing in r)]

    print('There are %d dishes with %s as exact ingredient/s.' % (len(data_term), " and ".join(['"%s"' % str(x) for x in ingred_list])))
    if len(data_term) > 0:
        print('Here is a look at some:')
        display(data_2.loc[data_term.index].head())

    print('There are %d dishes with a type/s of %s as ingredient/s.' % (len(data_word), " and ".join([str(x) for x in ingred_list])))
    if len(data_word) > 0:
        print('Here is a look at some:')
        display(data_2.loc[data_word.index].head())
        for ing in ingred_list:
            ingred_list = set(sum([x.split(',') for x in list(data_2.loc[data_word.index]['ingredients'])], []))
            ingred_list = [x.strip() for x in ingred_list if ing in x.strip()]
            print('\nList of ingredients with "%s" in it: %s' % (ing, ", ".join(['"%s"' % str(x) for x in ingred_list])))
        print("More specific?")


if __name__ == '__main__':
    filter_func(ingredients=['oil', 'rice'], diet='vegetarian')
    print('a')
