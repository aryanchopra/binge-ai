import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
from pathlib import Path

root= Path(".")
filepath = "sentiment.pkl"
clf = pickle.load(open(filepath, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def similarity_score():
    df = pd.read_csv('main_data.csv')
    cv = CountVectorizer() #count matrix
    count_matrix = cv.fit_transform(df['comb'])
    similarity = cosine_similarity(count_matrix) # creating a similarity score matrix
    return df,similarity

def recommend(movie):
    movie = movie.lower()
    try:
        df.head()
        similarity.shape
    except:
        df, similarity = similarity_score()
    if movie not in df['movie_title'].unique():
        return('Sorry! The movie searched does not exist in our database. Try another one.')
    else:
        i = df.loc[df['movie_title']==movie].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] 
        recoms = []
        for i in range(len(lst)):
            a = lst[i][0]
            recoms.append(df['movie_title'][a])
        return recoms   
def str_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = recommend(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommendations",methods=["POST"])
def recommendations():
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    suggestions = get_suggestions()

    rec_movies = str_to_list(rec_movies)
    rec_posters = str_to_list(rec_posters)
    cast_names = str_to_list(cast_names)
    cast_chars = str_to_list(cast_chars)
    cast_profiles = str_to_list(cast_profiles)
    cast_bdays = str_to_list(cast_bdays)
    cast_bios = str_to_list(cast_bios)
    cast_places = str_to_list(cast_places)
    
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'html.parser')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments 
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Positive' if pred else 'Negative')

    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)
