# Welcome to Cocktail Recommender! 

Cocktail Recommender is an NLP model created to ease the difficulty of settling on a drink choice from a plethora of options, in order to optimize satisfaction. 
The system uses **Word2Vec** to understand relationships between ingredients and suggest drinks with similar flavor profiles.

## Features
- Load and process cocktail recipes from a CSV file.
- Tokenize and vectorize ingredients using **gensim Word2Vec**.
- Recommend top cocktails based on similarity to user-provided ingredients.
- Multiple recommendation approaches (pairwise similarity, cosine similarity).
- Customizable number of recommendations.

## Installation

1. **Clone this repository**
```bash
git clone https://github.com/larisarod/cocktail-recommender.git
cd cocktail-recommender

pip install pandas gensim scikit-learn #install dependencies

python copy_of_final_project.py
```
2. **EXAMPLE USAGE**

EXAMPLE INPUT:
user_ingredients = ['Gin', 'Lime Juice']

EXAMPLE OUTPUT:
           name                           ingredients  similarity
0  Gimlet       ['Gin', 'Lime Juice', 'Simple Syrup']   0.9231
1  Tom Collins  ['Gin', 'Lemon Juice', 'Sugar', ...]   0.8914
