#IMPORT LIBRARIES
import pickle
from flask import Flask, render_template, request
import requests
import pandas as pd
from scipy import sparse
import pyrebase
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

#LOAD THE KNN, ITEM-ITEM MODELS AND VECTORIZER FROM DISK - PKL
corrMatrix = pickle.load(open('corrMatrix.pkl','rb'))
pivot_table = pickle.load(open('pivotKNN.pkl','rb'))
movies = pickle.load(open('movies.pkl','rb'))

#LOAD DATASETS - CSV
links = pd.read_csv('dataset/links.csv')
movieLens = pd.read_csv('dataset/movies.csv')
tmdb_movies = pd.read_csv('datasets/tmdb_5000_credits.csv')
clean_coll = pd.read_csv('dataset/coll_clean_data.csv')

#FIREBASE SCRIPT
firebaseScript = {
    "apiKey": "AIzaSyCF1TFsjaWZfGIRwF4frvoVMNc66uVyIjs",
    "authDomain": "movie-recommender-c49e8.firebaseapp.com",
    "databaseURL": "https://movie-recommender-c49e8-default-rtdb.firebaseio.com",
    "projectId": "movie-recommender-c49e8",
    "storageBucket": "movie-recommender-c49e8.appspot.com",
    "messagingSenderId": "642745412970",
    "appId": "1:642745412970:web:a28785b7d5fe090be1c913",
    "measurementId": "G-JE6L3EKLQG"
}
fb = pyrebase.initialize_app(firebaseScript)
auth = fb.auth()
storage = fb.database()

my_api = "0826c0f9ab2e9bbfd6308fc84784e24f"

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

#FUNCTION GIVING COMPLETE MOVIE DETAILS BY MOVIE ID
def movie_det(movie_id):
    c_url = "https://api.themoviedb.org/3/movie/{}/credits?api_key=".format(movie_id)
    cast_url = c_url + my_api + "&language=en-US"
    u = "https://api.themoviedb.org/3/movie/{}?api_key=".format(movie_id)
    url = u + my_api + "&append_to_response=videos"
    val = requests.get(url) # getting data from API request
    val = val.json() # converting to JSON
    val2 = requests.get(cast_url) # getting data from API request
    val2 = val2.json() # converting to JSON
    mov_cast = val2['cast']
    cast_pic = [] # list of cast photos
    cast_name = [] # list of cast names
    counter = 0
    for j in mov_cast:
        if counter > 11:
            break
        cast_pic_path = j['profile_path']
        cast_pic.append("https://image.tmdb.org/t/p/w500" + str(cast_pic_path))
        cast_name_path = j['name']
        cast_name.append(cast_name_path)
        counter=counter+1
    poster_path_url = val['poster_path']
    back_path_url = val['backdrop_path']
    full_pospath_url = "https://image.tmdb.org/t/p/w500" + poster_path_url
    full_backpath_url = "https://image.tmdb.org/t/p/w500" + back_path_url
    mov_year_det = val['release_date']
    mov_time_det = val['runtime']
    mov_overview_det = val['overview']
    mov_trail = val['videos']['results']
    mov_trail2 = mov_trail[0]
    mov_trail3 = mov_trail2['key']
    if(mov_trail2['site'] == "YouTube"):
        full_vid_path = 'https://www.youtube.com/watch?v='+mov_trail3
    else:
        full_vid_path = 'https://vimeo.com/'+mov_trail3
    genres_det = []
    for i in val['genres']:
        genres_det.append(i['name'])
    return full_pospath_url,full_backpath_url,mov_year_det,mov_time_det,mov_overview_det,genres_det,full_vid_path,cast_pic,cast_name

#FUNCTION GIVING PARTIAL MOVIE DETAILS BY MOVIE ID
def fetch_poster(movie_id):
    u = "https://api.themoviedb.org/3/movie/{}?api_key=".format(movie_id)
    url = u + my_api + "&language=en-US"
    data = requests.get(url) # getting data from API request
    data = data.json() # converting to JSON
    poster_path = data['poster_path']
    if type(poster_path) != str:
        poster_path = "ff"
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    mov_year = data['release_date']
    mov_time = data['runtime']
    mov_vote = data['vote_average']
    full_name = data['title']
    return full_path,mov_year,mov_time,mov_vote,full_name

#FUNCTION GIVING CONTENT BASED RECOMMENDATIONS AND THEIR DETAILS BY MOVIE TITLE THAT SEARCHED/SELECTED BY USER
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = [] # list of names of recommended movies by content based recommendation
    recommended_movie_posters = [] # list of posters of recommended movies by content based recommendation
    released_movie = [] # list of released year of recommended movies by content based recommendation
    movie_dur = [] # list of duration of recommended movies by content based recommendation
    vote_mov = [] # list of avg rating of recommended movies by content based recommendation
    for i in distances[1:9]:
        movie_id = tmdb_movies.iloc[i[0]].movie_id
        full_path,mov_year,mov_time,mov_vote,full_name = fetch_poster(movie_id)
        recommended_movie_posters.append(full_path)
        recommended_movie_names.append(movies.iloc[i[0]].title)
        released_movie.append(mov_year)
        movie_dur.append(mov_time)
        vote_mov.append(mov_vote)
    return recommended_movie_names,recommended_movie_posters,released_movie,movie_dur,vote_mov

#FUNCTION GIVING DETAILS OF MOVIE PRESENT IN MOVIE ARRAY WHICH GOT BY COLLABORATIVE FILTERING (ITEM-ITEM BASED)
def coll_det(movieArray):
    recommended_movie_names = [] # list of names of recommended movies by collaborative filtering (item-item)
    recommended_movie_posters = [] # list of posters of recommended movies by collaborative filtering (item-item)
    released_movie = [] # list of released year of recommended movies by collaborative filtering (item-item)
    movie_dur = [] # list of duration of recommended movies by collaborative filtering (item-item)
    vote_mov = [] # list of avg rating of recommended movies by collaborative filtering (item-item)
    for i in movieArray:
        if i not in movieLens['title'].values:
            break
        lensId = movieLens.loc[movieLens['title'] == i, 'movieId']
        lensId = lensId.iloc[0]
        lensId2 = links.loc[links['movieId'] == lensId, 'tmdbId']
        lensId2 = int(lensId2.iloc[0])
        full_path,mov_year,mov_time,mov_vote,full_name = fetch_poster(lensId2)
        recommended_movie_posters.append(full_path)
        recommended_movie_names.append(full_name)
        released_movie.append(mov_year)
        movie_dur.append(mov_time)
        vote_mov.append(mov_vote)
    return recommended_movie_names,recommended_movie_posters,released_movie,movie_dur,vote_mov

#FUNCTION GIVING SIMILAR RATINGS CORRESPONDING TO MOVIE THAT RATED BY USER
def collaborative(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings

#FUNCTION GIVING MOVIES ARRAY SIMILAR TO MOVIES RATED BY USER
def item_item(dyn_rating):
    similar_movies = pd.DataFrame()
    for i in dyn_rating:
        for k, v in i.items():
            similar_movies = similar_movies.append(collaborative(k,int(v)),ignore_index = True)
    val = similar_movies.sum().sort_values(ascending=False).head(21)
    return val.index

#FUNCTION OF KNN BASED RECOMMENDATION
def knn(movTitle):
    features_matrix=csr_matrix(pivot_table.values)
    model=NearestNeighbors(metric='cosine',algorithm='brute')
    model.fit(features_matrix)
    distances,indices=model.kneighbors(pivot_table.loc[movTitle,:].values.reshape(1,-1),n_neighbors=9)
    ansArr=pivot_table.iloc[indices.flatten()].index
    return ansArr

# get movie suggestions for auto complete
def get_suggestions():
    return list(movies['title'].str.capitalize())


app = Flask(__name__)

suggestions = get_suggestions()

@app.route("/")
def home():
    return render_template("register.html")

@app.route("/register" , methods=['POST','GET'])
def signup():
    email = request.form["email"]
    password = request.form["password"]
    try:
        user = auth.create_user_with_email_and_password(email, password)
        return render_template("login.html")
    except:
        return ("E mail already exist")
        
@app.route("/login" , methods = ['POST'])
def login():
    email = request.form["email"]
    password = request.form["password"]
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return render_template("index.html",suggestions=suggestions)
    except:
        return ("Register")

@app.route("/recommend", methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html',suggestions=suggestions)
    if request.method == 'POST':
        selected_movie = request.form['moviename'].title()
        movie_list = movies['title'].values
        if selected_movie not in movie_list:
            return render_template('notFound.html', selected_movie=selected_movie)
        else:
            indi = movies[movies['title'] == selected_movie].index[0]
            movie_sel_id = movies[movies['title'] == selected_movie].movie_id[indi]
            full_pospath_url,full_backpath_url,mov_year_det,mov_time_det,mov_overview_det,genres_det,full_vid_path,cast_pic,cast_name = movie_det(movie_sel_id)
            recommended_movie_names,recommended_movie_posters,released_movie,movie_dur,vote_mov = recommend(selected_movie)
            if movie_sel_id in links['tmdbId'].values:
                movieLensId = links.loc[links['tmdbId'] == movie_sel_id, 'movieId']
                movieLensId = movieLensId.iloc[0]
                if (movieLensId in movieLens['movieId'].values) and (movieLensId in clean_coll['movieId'].values):
                    val2 = movieLens.loc[movieLens['movieId'] == movieLensId, 'title']
                    val2 = val2.iloc[0]
                    knnArray = knn(val2)[1:] # excluding first movie since it is the requested movie itself
                else:
                    knnArray = [] # list of recommended movies by KNN algorithm
            else:
                knnArray = [] # list of recommended movies by KNN algorithm
            knn_movie_names,knn_movie_posters,knn_released_movie,knn_movie_dur,knn_vote_mov = coll_det(knnArray)
            # passing all the data to the movie-details html file
            return render_template("movie-details.html", recommended_movie_names=recommended_movie_names, recommended_movie_posters=recommended_movie_posters,released_movie=released_movie,movie_dur=movie_dur,vote_mov=vote_mov,length=len(recommended_movie_names),
            full_pospath_url=full_pospath_url,full_backpath_url=full_backpath_url,mov_year_det=mov_year_det,mov_time_det=mov_time_det,mov_overview_det=mov_overview_det,genres_det=genres_det,full_vid_path=full_vid_path,sel_title=selected_movie,cast_pic=cast_pic,cast_name=cast_name,length2=len(cast_name),
            knn_movie_names=knn_movie_names,knn_movie_posters=knn_movie_posters,knn_released_movie=knn_released_movie,knn_movie_dur=knn_movie_dur,knn_vote_mov=knn_vote_mov,length3=len(knn_movie_names))

@app.route("/index/<val>",methods=['GET', 'POST'])
def got_rate(val):
    if request.method == 'POST':
        new_val = request.form["temp"]
        user_uid = auth.current_user['localId']
        indi = movies[movies['title'] == val].index[0]
        movie_sel_id = movies[movies['title'] == val].movie_id[indi]
        if movie_sel_id not in links['tmdbId'].values:
            return render_template('index.html',suggestions=suggestions)
        movieLensId = links.loc[links['tmdbId'] == movie_sel_id, 'movieId']
        movieLensId = movieLensId.iloc[0]
        if movieLensId not in movieLens['movieId'].values:
            return render_template('index.html',suggestions=suggestions)
        val2 = movieLens.loc[movieLens['movieId'] == movieLensId, 'title']
        val2 = val2.iloc[0]
        storage.child(user_uid).push({val2:new_val})
        v_val = storage.child(user_uid).get()
        v_val = v_val.val()
        return render_template("index.html",suggestions=suggestions)

@app.route("/my-rec")
def my_rec():
    user_uid = auth.current_user['localId']
    movRate = storage.child(user_uid).get()
    dyn_rating = movRate.val().values()
    itemArray = item_item(dyn_rating)[1:] # excluding first movie since it is the requested movie itself
    recommended_movie_names,recommended_movie_posters,released_movie,movie_dur,vote_mov = coll_det(itemArray)
    print("u")
    # passing all the data to the my-rec html file
    return render_template("my-rec.html", recommended_movie_names=recommended_movie_names,recommended_movie_posters=recommended_movie_posters,released_movie=released_movie,movie_dur=movie_dur,vote_mov=vote_mov,length=len(recommended_movie_names))

if __name__ == "__main__":
    app.run(debug=True)