# Import necesary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
ps = PorterStemmer()


def vectorize_ingred(food_df):
    ingred_set = set(sum(food_df['ingredients'], []))
    ingred_one_hot = pd.DataFrame(0, index=food_df.index, columns=ingred_set)
    for i, row in food_df.iterrows():
        for ing in row['ingredients']:
            ingred_one_hot.loc[i, ing] = 1
    return ingred_one_hot


data = pd.read_csv('../indian_food.csv', header=0)
print('Dataset size is %s' % str(data.shape))
print('Dataset columns are %s' % list(data.columns))

# Lower casing all values and handling typos
for col_name in ['name', 'ingredients', 'diet', 'flavor_profile', 'course', 'state', 'region']:
    data[col_name] = data[col_name].str.lower()
data['ingredients'] = data['ingredients'].str.replace('chilli', 'chili')
data['ingredients'] = data['ingredients'].str.replace('yoghurt', 'yogurt')
# display(data.head())

# Replacing -1 values with nan
data[(data == -1) | (data == '-1')] = np.nan
print('Columns %s has missing (NA) values.' % list(data.columns[data.isna().any()]))
data_print = data.apply(lambda x: x.astype(str).str.title())

# Handling nan values
data_print = data.copy()
# display(data_print.tail())
print('Mean of prep and cook time: \n' + str(data.mean().round()))
data_2 = data.fillna(data.mean().round())
for column in data_2.columns:
    data_2[column].fillna(data[column].mode()[0], inplace=True)

# Creating the dataframe for printing purposes. Replaced missing values shown with an asterick
data_print = data_print.fillna(data_2.astype(str) + '*')
data_print = data_print.apply(lambda x: x.astype(str).str.title())
# display(data_print.tail())

data_term_gf = data_2.copy()
gluten_ing = ['wheat', 'gram-flour', 'barley', 'yeast', 'bulgur', 'durum', 'kamut', 'malt', 'matzo', 'oat', 'rye', 'semolina', 'gram flour']
indexes_to_drop = []
for i, row in data_term_gf.iterrows():
    for g_ing in gluten_ing:
        if g_ing in row['ingredients']:
            indexes_to_drop.append(i)
            break

data_term_gf['ingredients'] = data_term_gf['ingredients'].apply(lambda x: [w.strip() for w in x.split(',')])
data_term_gf['ingredients'] = data_term_gf['ingredients'].apply(lambda x: [term.replace(' ', '-') for term in x])
for g_ing in gluten_ing:
    data_term_gf['ingredients'] = data_term_gf['ingredients'].apply(lambda row: [x for x in row if g_ing not in x])

term_remove_list = ['water', 'salt', 'pepper', 'sweet']
data_term_gf['ingredients'] = data_term_gf['ingredients'].apply(lambda row: [x for x in row if x not in term_remove_list])

term_replace_dic = {'maida': 'maida-flour', 'powdered-sugar': 'sugar', 'whole-egg': 'egg'}
for i, row in data_term_gf.iterrows():
    for ing, ing_correct in term_replace_dic.items():
        if ing in row['ingredients']:
            row['ingredients'].remove(ing)
            row['ingredients'].append(ing_correct)

word_replace_dic = {'-and': '', 'frozen-': '', 'canned-': '', '-powder': ''}
for k, v in word_replace_dic.items():
    data_term_gf['ingredients'] = data_term_gf['ingredients'].apply(lambda row: [x.replace(k, v) for x in row])

# Stemming the ingredient terms
data_term_gf['ingredients'] = data_term_gf['ingredients'].apply(lambda row: [ps.stem(x) for x in row])

data_word_gf = data_term_gf.copy()
data_word_gf['ingredients'] = data_word_gf['ingredients'].apply(lambda x: list(sum([s.split('-') for s in x], [])))

# Generating the ingredient bag of the TERMS and WORDS vectorizing in a one-hot fashion
term_one_hot_gf = vectorize_ingred(data_term_gf)
word_one_hot_gf = vectorize_ingred(data_word_gf)

# Similarity matrices based on terms, words and additional data using Cosine-similarity methods
term_sim_mat_gf = cosine_similarity(np.array(term_one_hot_gf))
term_sim_df_gf = pd.DataFrame(data=term_sim_mat_gf, index=data_term_gf.index, columns=data_term_gf['name'])

word_sim_mat_gf = cosine_similarity(np.array(word_one_hot_gf))
word_sim_df_gf = pd.DataFrame(data=word_sim_mat_gf, index=data_word_gf.index, columns=data_word_gf['name'])

# Vectorizing diet and course fields ina one-hot fashion and creating one-hot array
diet_one_hot = pd.get_dummies(data_2.diet, prefix='diet')
course_one_hot = pd.get_dummies(data_2.course, prefix='course')
flavor_profile_one_hot = pd.get_dummies(data_2.flavor_profile, prefix='flavor_profile')
state_one_hot = pd.get_dummies(data_2.state, prefix='state')
region_one_hot = pd.get_dummies(data_2.region, prefix='region')
misc_one_hot = pd.concat([diet_one_hot, course_one_hot, flavor_profile_one_hot, state_one_hot, region_one_hot], axis=1)
misc_sim_mat = cosine_similarity(np.array(misc_one_hot))
misc_sim_df_gf = pd.DataFrame(data=misc_sim_mat, index=data_2.index, columns=data_2['name'])

term_w_gf = 0.48
word_w_gf = 0.47
data_w_gf = 0.05


def recommend_similar_dish_gluten_free(dish_name, diet=None, max_prep_time=None, max_cook_time=None, course=None):
    """
    Recommend a Gluten-free food from the indian food dataset based on similarity to the dish_name parameter.
    The similarities are measured using cosine similarity method based on three criterias of food ingredient terms,
    food ingredient words, foods additional data (origin). The foods are them sorted from most similar to least
    similar. Similar foods data frame can be filtered based on 4 criteria of diet type, course, maximum preparation
    and cooking time.

    At fist all input are checked to be of the right format and within the acceptable values. This process is not
    100% error proof but can prevent elementary input errors. Below examples are all correct application of function.
    recommend_similar_dish(dish_name='Bora Sawul', max_prep_time=100, max_cook_time=100, diet='non vegetarian')
    recommend_similar_dish(dish_name='Goja', max_prep_time=100, max_cook_time=100, diet='vegetarian', course='snack')
    recommend_similar_dish(dish_name='Shukto')

    :param dish_name: The dish name against which similarities of other dishes are measured
    :param diet: The diet to be 'vegetarian' or 'non vegetarian'
    :param course: The course for which the dish should be served can be 'dessert', 'main course', 'starter', 'snack'
    :param max_prep_time: The maximum preparation rime
    :param max_cook_time: The maximum cooking time
    :return: Sorted dish data frame
    """
    global data_print
    global term_sim_df_gf
    global word_sim_df_gf
    global misc_sim_df_gf
    if dish_name is None or str(dish_name).lower() not in set(term_sim_df_gf.columns):
        print('Provided dish name "%s" is not available in Indian Dish Dataset. Please try again.' % dish_name)
    dish_name = dish_name.lower()
    data_sorted = pd.concat([data_print, term_sim_df_gf[dish_name]], axis=1)
    data_sorted.rename(columns={dish_name: 'term score'}, inplace=True)

    data_sorted = pd.concat([data_sorted, word_sim_df_gf[dish_name]], axis=1)
    data_sorted.rename(columns={dish_name: 'word score'}, inplace=True)

    data_sorted = pd.concat([data_sorted, misc_sim_df_gf[dish_name]], axis=1)
    data_sorted.rename(columns={dish_name: 'misc score'}, inplace=True)

    # Generating the total similarity score based on which dish data frame is sorted
    data_sorted['similarity score'] = term_w_gf * data_sorted['term score'] + word_w_gf * data_sorted[
        'word score'] + data_w_gf * data_sorted['misc score']
    data_sorted.sort_values(by='similarity score', ascending=False, inplace=True)

    # Filtering based on diet
    if diet is not None:
        if diet.lower() not in data_term_gf['diet'].unique():
            print('Diet option input "%s" not valid. You may choose from: %s' % (diet, data_2_gf['diet'].unique()))
            return
        else:
            print('Filtering similar dishes based on diet type of %s.' % diet)
            data_sorted = data_sorted[data_term_gf['diet'] == diet.lower()]

    # Filtering based on course
    if course is not None:
        if course.lower() not in data_term_gf['course'].unique():
            print(
                'Course option input "%s" not valid. You may choose from: %s' % (course, data_2_gf['course'].unique()))
            return
        else:
            print('Filtering similar dishes based on course type of %s.' % course)
            data_sorted = data_sorted[data_term_gf['course'] == course.lower()]

    # Filtering based on maximum preparation time
    if max_prep_time is not None:
        try:
            max_prep_time = int(max_prep_time)
            print('Filtering similar dishes with maximum preparation time of %d.' % max_prep_time)
            data_sorted = data_sorted[(data_term_gf['prep_time'] <= max_prep_time)]
        except ValueError:
            print('Incorrect input for prep time (%s)' % max_prep_time)
            return

    # Filtering based on maximum cooking time
    if max_cook_time is not None:
        try:
            max_cook_time = int(max_cook_time)
            print('Filtering similar dishes with maximum cooking time of %d.' % max_cook_time)
            data_sorted = data_sorted[(data_term_gf['cook_time'] <= max_cook_time)]
        except ValueError:
            print('Incorrect input for cook time (%s)' % max_cook_time)
            return

    print('_' * 80)
    if len(data_sorted) == 0:
        print('There are no similar dishes under mentioned criteria')
    else:
        data_sorted.drop(indexes_to_drop, inplace=True)
        print('Here are similar dishes to "%s" arranged from more similar to least similar' % dish_name)
        # data_res = pd.concat([data_print.loc[data_sorted.index], data_sorted['similarity score']], axis=1)
        display(data_sorted.head())
        return data_sorted


if __name__ == '__main__':
    # print('Regular recommendation')
    # recommend_similar_dish(dish_name='Kaju Katli')
    # print('\n' + '-' * 90)
    print('Gluten-free recommendation')
    recommend_similar_dish_gluten_free(dish_name='Kaju Katli')
    print('')
