import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
from IPython.display import display

ps = PorterStemmer()

data_2 = pd.read_csv('../indian_food.csv', header=0)
# data_2 = data.copy()

# Lower casing all values and handling typos
for col_name in ['name', 'ingredients', 'diet', 'flavor_profile', 'course', 'state', 'region']:
    data_2[col_name] = data_2[col_name].str.lower()
data_2['ingredients'] = data_2['ingredients'].str.replace('chilli', 'chili')
data_2['ingredients'] = data_2['ingredients'].str.replace('yoghurt', 'yogurt')
print('Dataset size is %s' % str(data_2.shape))
print('Dataset columns are %s' % list(data_2.columns))

# Replacing -1 values with nan
data_2[(data_2 == -1) | (data_2 == '-1')] = np.nan
print('Columns %s has missing (NA) values.' %list(data_2.columns[data_2.isna().any()]))


def plot_pie_chart(dataframe, col_name):
    col_count = dataframe[col_name].value_counts().reset_index()
    plt.pie(col_count[col_name], labels=col_count['index'], radius=1.2, autopct='%0.1f%%', explode=[0.1] * len(col_count['index']))
    plt.title("Percentage of Vegeterian vs Non-Vegeterian Dishes")
    plt.show()

# Handling nan values
data_print = data_2.copy()
print('Mean of prep and cook time: \n' + str(data_2.mean().round()))
data_2 = data_2.fillna(data_2.mean().round())
for column in data_2.columns:
    data_2[column].fillna(data_2[column].mode()[0], inplace=True)

data_print = data_print.fillna(data_2.astype(str) + '*')
data_print = data_print.apply(lambda x: x.astype(str).str.title())

data_term = data_2.copy()
# Tokenizing ingredients strings
data_term['ingredients'] = data_term['ingredients'].apply(lambda x: [w.lower().strip() for w in x.split(',')])
# Hyphenating combination ingredients
data_term['ingredients'] = data_term['ingredients'].apply(lambda x: [term.replace(' ', '-') for term in x])

## List of stop words to be removed
# Removing stop words and replacing words with typos or same concepts
term_remove_list = ['water', 'salt', 'pepper', 'sweet']
data_term['ingredients'] = data_term['ingredients'].apply(lambda row: [x for x in row if x not in term_remove_list])

# Dictionary of terms with typos or similar concept to be replaced
term_replace_dic = {'maida': 'maida-flour',
                    'powdered-sugar': 'sugar',
                    'whole-egg': 'egg'}
for i, row in data_term.iterrows():
    for ing, ing_correct in term_replace_dic.items():
        if ing in row['ingredients']:
            row['ingredients'].remove(ing)
            row['ingredients'].append(ing_correct)

# Dictionary of words to be replaced
word_replace_dic = {'-and': '',
                    'frozen-': '',
                    'canned-': '',
                    '-powder': ''}
for k, v in word_replace_dic.items():
    data_term['ingredients'] = data_term['ingredients'].apply(lambda row: [x.replace(k, v) for x in row])
# Removing "-powder" suffix meaning "garlic" and "garlic-powder" will be treated the same

# Stemming the ingredient terms
data_term['ingredients'] = data_term['ingredients'].apply(lambda row: [ps.stem(x) for x in row])

data_word = data_term.copy()
data_word['ingredients'] = data_word['ingredients'].apply(lambda x: list(sum([s.split('-') for s in x], [])))

def vectorize_ingred(food_df):
    ingred_set = set(sum(food_df['ingredients'], []))
    ingred_one_hot = pd.DataFrame(0, index=food_df.index, columns=ingred_set)
    for i, row in food_df.iterrows():
        for ing in row['ingredients']:
            ingred_one_hot.loc[i, ing] = 1
    return ingred_one_hot

# word_list = sum(data_word['ingredients'], [])
# word_set = set(word_list)
# from collections import Counter
# words_count = Counter(word_set)
# words_count = {k: v for k, v in sorted(words_count.items(), key=lambda item: -item[1])}

# Generating the ingredient bag of the TERMS and vectorizing in a one-hot fashion
word_one_hot = vectorize_ingred(data_word)
term_one_hot = vectorize_ingred(data_term)
# Generating the ingredient bag of the WORDS and vectorizing in a one-hot fashion

# Vectorizing diet and course fields ina one-hot fashion and creating one-hot array
diet_one_hot = pd.get_dummies(data_2.diet, prefix='diet')
course_one_hot = pd.get_dummies(data_2.course, prefix='course')
flavor_profile_one_hot = pd.get_dummies(data_2.flavor_profile, prefix='flavor_profile')
state_one_hot = pd.get_dummies(data_2.state, prefix='state')
region_one_hot = pd.get_dummies(data_2.region, prefix='region')
misc_one_hot = pd.concat([diet_one_hot, course_one_hot, flavor_profile_one_hot, state_one_hot, region_one_hot], axis=1)


# Similarity Heatmap based on "diet", "course" and "ingredients" fields using Cosine-similarity methods
term_sim_mat = cosine_similarity(np.array(term_one_hot))
term_sim_df = pd.DataFrame(data=term_sim_mat, index=data_term.index, columns=data_term['name'])

word_sim_mat = cosine_similarity(np.array(word_one_hot))
word_sim_df = pd.DataFrame(data=word_sim_mat, index=data_word.index, columns=data_word['name'])

misc_sim_mat = cosine_similarity(np.array(misc_one_hot))
misc_sim_df = pd.DataFrame(data=misc_sim_mat, index=data_2.index, columns=data_2['name'])


def vectorize_ingred(food_df):
    ingred_set = set(sum(food_df['ingredients'], []))
    ingred_one_hot = pd.DataFrame(0, index=food_df.index, columns=ingred_set)
    for i, row in data_term.iterrows():
        for ing in row['ingredients']:
            ingred_one_hot.loc[i, ing] = 1


def recommend_similar_dish(dish_name, diet=None, max_prep_time=None, max_cook_time=None, course=None):
    """
    Recommend a food from the indian food dataset based on similarity to the dish_name parameter.
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
    global term_sim_df
    global word_sim_df
    global misc_sim_df
    if dish_name is None or str(dish_name).lower() not in set(term_sim_df.columns):
        print('Provided dish name "%s" is not available in Indian DIsh Dataset. Please try again.' % dish_name)
    data_sorted = pd.concat([data_term, term_sim_df[dish_name]], axis=1)
    data_sorted.rename(columns={dish_name: 'term score'}, inplace=True)

    data_sorted = pd.concat([data_sorted, word_sim_df[dish_name]], axis=1)
    data_sorted.rename(columns={dish_name: 'word score'}, inplace=True)

    data_sorted = pd.concat([data_sorted, misc_sim_df[dish_name]], axis=1)
    data_sorted.rename(columns={dish_name: 'misc score'}, inplace=True)

    # Generating the total similarity score based on which dish data frame is sorted
    data_sorted['similarity score'] = 0.5 * data_sorted['term score'] + 0.5 * data_sorted['word score'] + 0.05 * \
                                      data_sorted['misc score']
    data_sorted.sort_values(by='similarity score', ascending=False, inplace=True)

    # Filtering based on diet
    if diet is not None:
        if diet.lower() not in data_2['diet'].unique():
            print('Diet option input "%s" not valid. You may choose from: %s' % (diet, data_2['diet'].unique()))
            return
        else:
            print('Filtering similar dishes based on diet type of %s.' % diet)
            data_sorted = data_sorted[data_2['diet'] == diet.lower()]

    # Filtering based on course
    if course is not None:
        if course.lower() not in data_2['course'].unique():
            print('Course option input "%s" not valid. You may choose from: %s' % (course, data_2['course'].unique()))
            return
        else:
            print('Filtering similar dishes based on course type of %s.' % course)
            data_sorted = data_sorted[data_2['course'] == course.lower()]

    # Filtering based on maximum preparation time
    if max_prep_time is not None:
        try:
            max_prep_time = int(max_prep_time)
            print('Filtering similar dishes with maximum preparation time of %d.' % max_prep_time)
            data_sorted = data_sorted[(data_2['prep_time'] <= max_prep_time)]
        except ValueError:
            print('Incorrect input for prep time (%s)' % max_prep_time)
            return

    # Filtering based on maximum cooking time
    if max_cook_time is not None:
        try:
            max_cook_time = int(max_cook_time)
            print('Filtering similar dishes with maximum cooking time of %d.' % max_cook_time)
            data_sorted = data_sorted[(data_2['cook_time'] <= max_cook_time)]
        except ValueError:
            print('Incorrect input for cook time (%s)' % max_cook_time)
            return

    print('_' * 80)
    if len(data_sorted) == 0:
        print('There are no similar dishes under mentioned criteria')
    else:
        print('Here are similar dishes to "%s" arranged from more similar to least similar' % dish_name)
        data_res = pd.concat([data_print.loc[data_sorted.index], data_sorted['similarity score']], axis=1).iloc[1:]
        display(data_res.head())
        return data_sorted


if __name__ == '__main__':
    a = recommend_similar_dish(dish_name='sohan papdi')
    print('a')
