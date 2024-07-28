import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

movies_data = pd.read_csv('movies.csv')

# selecting the relevent features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']

# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
    
# combining all the selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

# converting the text data to feature vector
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# getting the movie name from user
movie_name = input('Your favourite movie name:')

# crating a list with all the movie name given in the dataset
list_of_all_titles = movies_data['title'].tolist()
print('list of all titles:...',list_of_all_titles)

# finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name,list_of_all_titles)
print('find close match:...',find_close_match)

close_match = find_close_match[0]
print('close match:...',close_match)

# find the index of the movie of the title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print('index of the movie:...',index_of_the_movie)

# getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
print('similarity score:...',similarity_score)
print('len of similirity score:...',len(similarity_score))

# sorting the movies based on their similirty score
sorted_similar_movies = sorted(similarity_score,key=lambda x:x[1],reverse=True)
print('sorted similar movies:...',sorted_similar_movies)

# print the name of similar movies based on the index
print('Movie suggested for you:\n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i<=30:
        print(i,'.',title_from_index)
        i+=1
