import pandas as pd
import numpy as np
from langdetect import detect
from autocorrect import Speller

data= pd.read_csv('dataset.csv')
data.drop(['Unnamed: 0','track_id'], axis=1, inplace=True)

def correct_text(text):
    try:
        lang = detect(text)
        spell = Speller(lang)
        corrected_text = spell(text)
        return corrected_text
    
    except:
        return text

# Correction of the words if exists
data['artists']= correct_text(data['artists'])
data['album_name']= correct_text(data['album_name'])
data['track_name']= correct_text(data['track_name'])

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Encoding categorical variable into numeric
label_encoder = LabelEncoder()
data['track_genre_encoded'] = label_encoder.fit_transform(data['track_genre'])
data.drop(['track_genre'], axis=1, inplace=True)

numeric_columns = [
    'danceability', 'energy','loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'tempo','track_genre_encoded'
]

# Scaling to normalize numeric columns
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# Features for KNN model
x= data[['danceability', 'energy','loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'tempo','track_genre_encoded']]
y= data['track_name']
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.2)

# Model Fit
knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean')
knn_model.fit(x_train.values,y_train)

from fuzzywuzzy import process

# Song Recommendation
recommendation = x_test.merge(data[['track_name', 'album_name']], left_index=True, right_index=True)

def recommender(input_track_name, data, model):
    match = process.extractOne(input_track_name, recommendation['track_name'])
    idx = match[2]
    print()
    print('Song Selected:  ', recommendation['track_name'][idx])
    print()

    requiredSongFeatures = data.loc[idx, numeric_columns].values.reshape(1, -1)
    distances, indices = model.kneighbors(requiredSongFeatures)

    print('Recommended Songs')
    print()
    for i in indices[0]:
        print(data['track_name'][i])

input_track_name = input("Enter track name: ")
recommender(input_track_name, data, knn_model)