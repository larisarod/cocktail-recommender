# -*- coding: utf-8 -*-

import pandas as pd
from google.colab import files

uploaded = files.upload()

df = pd.read_csv('final_cocktails.csv')
df.head()

#tokenize ingredients
df['ingredients_list'] = df['ingredients'].apply(lambda x: eval(x) if isinstance(x, str) else []) #ingredients to list
df['ingredients_list'] = df['ingredients_list'].apply(lambda x: [item.lower() for item in x]) #lowercase
data = df['ingredients_list'].tolist() #tokenize
data

import gensim
from gensim.models import Word2Vec
model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100, window = 5, sg=1) #build/ train skip-gram model
print(f'Corpus Size: {model.corpus_total_words}')
print(f'Training Time: {model.total_train_time}')
print(f'Sample Words: {list(model.wv.index_to_key[:10])}')


words = model.wv.index_to_key
print(words) #check

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vector_list = model.wv[words] #vectorize words
vector_list #check vectorized words


def recommend_by_ingredients(user_ingredients, df, model, is_alcoholic=True, top_n=5):
    #recommend by similarity of ingredients user prefers and ingredients in drink
    def compute_similarity(drink_ingredients, user_ingredients, model):
        similarities = []
        for drink_ingredient in drink_ingredients:
            for user_ingredient in user_ingredients:
                if drink_ingredient in model.wv and user_ingredient in model.wv:
                    similarities.append(model.wv.similarity(drink_ingredient, user_ingredient)) #compute similarity and add to list
        return sum(similarities) / len(similarities) if similarities else 0 #avg similarity

    # filter based on the alcoholic parameter
    if is_alcoholic:
        filtered_df = df[df['alcoholic'].isin(['Alcoholic', 'Optional alcohol'])]
    else:
        filtered_df = df[df['alcoholic'].isin(['Non alcoholic', 'Optional alcohol'])]


    filtered_df['similarity'] = filtered_df['ingredients_list'].apply(
        lambda drink_ingredients: compute_similarity(drink_ingredients, user_ingredients, model) #similarity for all drinks
    )

    # sort drinks by similarity and return top 5 recommendations
    recommendations = filtered_df.sort_values(by='similarity', ascending=False)
    return recommendations[['name', 'ingredients', 'ingredientMeasures', 'instructions', 'similarity', 'drinkThumbnail']].head(top_n)

#testing code
user_ingredients = ['vodka']
recommend_by_ingredients(user_ingredients, df, model, is_alcoholic=True)

from sklearn.metrics.pairwise import cosine_similarity

#flavor profiles
flavor_profiles = {
    'sour': ['lemon juice', 'lime juice', 'orange juice'],
    'sweet': ['sugar', 'simple syrup', 'grenadine', 'honey'],
    'spicy': ['pepper', 'ginger'],
    'salty': ['salt', 'olives', 'pickle juice', 'soy sauce'],
    'fruity':['strawberry', 'banana', 'cherry', 'fruit juice']
}

def categorize_ingredient(ingredient, flavor_profiles, model):
    # ensure the ingredient is in the w2v vocabulary
    if ingredient not in model.wv.key_to_index:
        return 'unknown'

    max_similarity = -1
    best_category = 'unknown'

    # compare the ingredient to each flavor profile
    for category, profile_ingredients in flavor_profiles.items():
        for flavor_ingredient in profile_ingredients:
            if flavor_ingredient in model.wv.key_to_index:
                similarity = model.wv.similarity(ingredient, flavor_ingredient)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_category = category

    return best_category


# categorize each ingredient in the ingredients list
df['ingredient_categories'] = df['ingredients_list'].apply(
    lambda ingredients: [categorize_ingredient(ing, flavor_profiles, model) for ing in ingredients]
)

# find the dominant flavor for each drink
from collections import Counter

def get_dominant_flavor(categories):
    if not categories:
        return 'unknown'
    category_counts = Counter(categories)
    return category_counts.most_common(1)[0][0]

df['dominant_flavor'] = df['ingredient_categories'].apply(get_dominant_flavor)

#compute similarity of each drink to a flavor profile
def compute_flavor_similarity(ingredients):
    valid_ingredients = [ing for ing in ingredients if ing in model.wv.key_to_index]
    if not valid_ingredients:
        return 0
    # compute cosine similarity for each ingredient vector with the profile vector
    similarities = [
        cosine_similarity(
            model.wv[ing].reshape(1, -1),
            profile_vector.reshape(1, -1)
        )[0][0]
        for ing in valid_ingredients
    ]

    # avg the similarities
    return sum(similarities) / len(similarities)

    #similarity calculation
    filtered_drinks['flavor_similarity'] = filtered_drinks['ingredients_list'].apply(compute_flavor_similarity)

    # recommend top drinks
    return filtered_drinks.sort_values(by='flavor_similarity', ascending=False).head(top_n)

def recommend_by_flavor(user_flavor, df, model, top_n=5, is_alcoholic=True):
    # filter drinks by dominant flavor
    filtered_drinks = df[df['dominant_flavor'] == user_flavor]

    # filter drinks by alcoholic preference
    if is_alcoholic:
        filtered_drinks = filtered_drinks[filtered_drinks['alcoholic'].isin(['Alcoholic', 'Optional alcohol'])]
    else:
        filtered_drinks = filtered_drinks[filtered_drinks['alcoholic'].isin(['Non alcoholic', 'Optional alcohol'])]

    # compute profile vector
    profile_ingredients = flavor_profiles.get(user_flavor, [])
    valid_profile_ingredients = [ing for ing in profile_ingredients if ing in model.wv.key_to_index]

    global profile_vector
    profile_vector = sum([model.wv[ing] for ing in valid_profile_ingredients]) / len(valid_profile_ingredients)

    # compute similarity of each drink to the flavor profile
    filtered_drinks = filtered_drinks.copy()
    filtered_drinks['flavor_similarity'] = filtered_drinks['ingredients_list'].apply(compute_flavor_similarity)

    # recommend top drinks
    return_df = filtered_drinks.sort_values(by='flavor_similarity', ascending=False).head(top_n)
    return return_df[['name', 'ingredients', 'ingredientMeasures', 'instructions','dominant_flavor', 'flavor_similarity', 'drinkThumbnail']]

# testing code
user_flavor = 'sour'
recommend_by_flavor(user_flavor, df, model, is_alcoholic=True)


from google.colab import drive
drive.mount('/content/drive')
