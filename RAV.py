
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import time
from IPython.display import display, HTML
from collections import Counter
import csv

def plot_review_words_influence(data, num_words = 20):
    '''function to plot the influence of certain words (after training a linear regression model)
    on rating as a bar plot
    Input :
    data : (pandas dataframe) containing the reviews data 
    num_words : (int) number of words in the plot 
    Returns nothing'''
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import string
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import linear_model
    
    text = list( data['review contents'] )
    rating = list( data['user rating'] )
    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,max_features= 1500 ,stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(text)
    features_words = np.array( [ str(w) for w in tfidf_vectorizer.get_feature_names() ]  )

    regr = linear_model.LinearRegression()
    regr.fit(tfidf.todense(), np.array(rating))
    feature_idx = np.argsort(regr.coef_)[::-1]

    good_words = feature_idx[:num_words]
    bad_words = feature_idx[-num_words:]
    plt.figure(figsize=(20,5))
    plt.bar(np.arange(num_words),regr.coef_[good_words])
    plt.rc('xtick', labelsize=15); plt.rc('ytick', labelsize=13) 
    plt.xticks(np.arange(num_words), features_words[good_words],rotation=60)
    plt.title('Most positively contributing words to review', fontsize = 17)
    plt.show()
    
    plt.figure(figsize=(20,5))
    plt.bar(np.arange(num_words),regr.coef_[bad_words])
    plt.rc('xtick', labelsize=15); plt.rc('ytick', labelsize=13) 
    plt.xticks(np.arange(num_words), features_words[bad_words],rotation=60)
    plt.title('Most Negatively contributing words to review', fontsize = 17)
    plt.show()

def movie_search(data):
    '''Function to get details of a movie title from the database, which is collected from user 
    Input : 
    title : (str) movie title'''
    print('-'*30+'Movie Search'+'-'*30)
    time.sleep(0.3)
    title = raw_input("Enter movie Name : ").lower()
    if title in list( data['primaryTitle'] ):
        temp = data[data['primaryTitle']==title ]
        if temp.shape[0]==1:
            display(temp)
        else :
            print('Available movies with that title name, ')
            display(temp[['primaryTitle','ReleaseYear','genres','directors']])
            year = int( input("Enter release year of the movie :") )
            if year in list( temp['ReleaseYear'] ):
                print('Movie Details')
                display( temp[temp['ReleaseYear'] == year] )
            else :
                print('Incorrect input !')

    else :
        temp = data.loc[[title in i for i in  list(data['primaryTitle'])] ]
        if temp.shape[0] == 0 :
            print('No movie with this title in our database !')
            return
        if temp.shape[0] == 1:
            print('Movie with that word in title : ')
            display(temp)
            return
        if temp.shape[0] > 10:
            temp = temp.sort_values(['numVotes'], ascending=False)
            temp = temp[:10]
        print('Few movies with that word in title')
        display(temp[['primaryTitle','ReleaseYear','genres','directors']] )
        title = raw_input("\nEnter movie Name : ").lower()
        temp = temp[temp['primaryTitle']==title]
        year = int( input("Enter release year of the movie :") )
        if year in list( temp['ReleaseYear'] ):
            print('\nMovie Details')
            display( temp[temp['ReleaseYear'] == year] )
        else :
            print('Incorrect input !')
            
def get_director_movies(movies,title, year = 0, hash1 = 1):
    '''function to get all movies by a director, given one of his movie titles
    Input Parameters : 
    movies (pandas dataframe) = contains the movie data
    title (str) = the name of the movie
    year (int) = year of movie release
    hash1 (binary) = if 1, the hash values of names released. if 0 primary titles released
    Output : List of all movie titles by the director of the input movie'''
    assert isinstance(title,str) and isinstance(year,(int,long)) 
    temp = movies[ movies['primaryTitle'] == title ]
    if year != 0  :
        temp = temp[temp['ReleaseYear'] == year]
        if temp.shape[0]>=1 :
            director = temp['directors'].iloc[0]
        else :
            return []
    else :
        director = movies[ movies['primaryTitle'] == title ].iloc[0]
    if director == '\n':
        return []
    col = hash1*'tconst' + (1-hash1)*'primaryTitle'
    return list( movies[ movies['directors']== director] [col] )

def get_genre_movies(movies, title, year = 0, per = 0.2, hash1 = 1):
    '''function to get all movies belonging to same genres as the input movie title, with similar ratings
    Input Parameters : 
    movies (pandas dataframe) = contains the movie data
    title (str) = the name of the movie
    year (int) = year of movie release
    per (float) = the neighborhood of ratings as fraction
    hash1 (binary) = if 1, the hash values of names released. if 0 primary titles released
    Output : List of all movies of same genres'''
    assert isinstance(title,str) and isinstance(year,(int,long)) 
    temp = movies[ movies['primaryTitle'] == title ]
    if year !=0 :
        temp = temp[temp['ReleaseYear'] == year]
    if temp.shape[0]==0 :
        return []
    genres =  temp['genres'].iloc[0]
    rating =  movies[ movies['primaryTitle'] == title ]['averageRating'].iloc[0]
    temp = movies[movies['genres']== genres ]
    temp = temp[ temp['averageRating']< (1+per)*rating ]
    temp = temp[ temp['ReleaseYear'] <= year+10]
    temp = temp[ temp['ReleaseYear'] >= year-10]
    col = hash1*'tconst' + (1-hash1)*'primaryTitle'
    return list( temp[ temp['averageRating']> (1-per)*rating ][col] )

def get_highestrated_movies(movies, num = 100, hash1 = 1):
    '''function to get highest rated (average score * numVotes) movies of all time
    Input Parameters : 
    movies (pandas dataframe) = contains the movie data
    num (int) = number of movies needed
    hash1 (binary) = if 1, the hash values of names released. if 0 primary titles released
    Output : List of highest rated movies'''
    assert isinstance(num,int) and (num>0)
    temp = pd.DataFrame( movies[['tconst','primaryTitle']] )
    temp['totalscore'] = pd.Series( movies.loc[:, 'averageRating']*movies.loc[:,'numVotes'] )
    temp = temp.sort_values(['totalscore'], ascending=False)
    temp = temp[:num]
    col = hash1*'tconst' + (1-hash1)*'primaryTitle'
    return list( temp[col])
        
def get_recommendation(movies, title, year1=0, num = 6):
    '''function to get a list of movie recommendations given a movie title 
    Input Parameters : 
    title (str) = the name of the movie
    year (int) = year of movie release
    num (int) = number of recommendations needed
    Output : List of all movies recommendations'''
    assert isinstance(title,str) and isinstance(num,int) and isinstance(year1,(int,long)) 
    list1 = get_director_movies(movies,title, year = year1)
    list2 = get_genre_movies(movies,title,year = year1)
    list3 = get_highestrated_movies(movies)
    temp = movies[movies['primaryTitle']==title]
    temp = temp[temp['ReleaseYear']==year1]
    tconst1 = list( temp['tconst'] )[0]
    
    # weights of director suggestions and genre suggestions respectively
    lam1 = 1; lam2 = 0.6; lam3 = 0.3
    
    temp = dict(zip(list1,[lam1]*len(list1)))
    for w in list2:
        if w in temp:
            temp[w] += lam2
        else:
            temp[w] = lam2
    for w in list3:
        if w in temp:
            temp[w] += lam3
        else:
            temp[w] = lam3
            
    if tconst1 in temp :
        temp[tconst1] = 0
        
    keys = sorted(temp.values(),reverse=True)
    count = 0
    recommend = []
    if num>len(temp.keys()):
        num = len(temp.keys())
    for w in temp :
        if temp[w] in keys[:num]:
            recommend.append(w) 
            count += 1
        if count == num:
            break
    out = movies[ movies['tconst'].map(lambda x: x in recommend) ]
    return out.loc[:, out.columns != 'tconst']

def name_based_recommender(movies):
    """Interface for title based recommendation
    parameters:
    data:(Pandas DataFrame)Data frame read from.tsv file containing all required information"""
    title = raw_input('Enter movie title : ')
    title = title.lower()
    
    if sum( [title in i for i in  list(movies['primaryTitle'])] ) > 0 :
        if title in list( movies['primaryTitle'] ) :
            print('Available movies with that title name, ')
            temp = movies[movies['primaryTitle']==title ]
            display(temp[['primaryTitle','ReleaseYear','genres','averageRating','numVotes','directors']])
        else :
            print('Movies with that word in title,')
            temp = movies.loc[[title in i for i in  list(movies['primaryTitle'])] ]
            if temp.shape[0] == 1:
                display(temp[['primaryTitle','ReleaseYear','genres','averageRating','numVotes','directors']])
                title = str( list(temp['primaryTitle'])[0] )
            elif temp.shape[0] > 10:
                temp = temp.sort_values(['numVotes'], ascending=False)
                temp = temp[:10]
                display(temp[['primaryTitle','ReleaseYear','genres','averageRating','numVotes','directors']])
                title = raw_input('Enter the movie title : ')
            else :
                print('No movie with that word in title! Try again ...')
                return
                
        time.sleep(0.2)
        if temp.shape[0] == 1:
            year = list(temp['ReleaseYear'])[0]
        else :
            year = int( raw_input('\nEnter movie release date : ') )
        
        all_years = set(temp['ReleaseYear'])
        if year not in all_years:
            print('Incorrect Release Year ! Try again...')
        else:
            num_movies = int(raw_input("Enter the number of recommendations you would like to see : "));
            df = get_recommendation(movies,title , year , num_movies) 
            print('Recommendations :')
            df['total'] = df.eval('averageRating * numVotes')
            df = df.sort_values('total',axis = 0,ascending=False)
            df = df.loc[:,df.columns != 'total']
            display(HTML(df.to_html()))
    else :
        print "\nThis title is not in our database. Try again ...."
    
def year_interface(data):
    """Heart of the year based recommender system. Extracts user preferences of year and the number of movies 
    within the year the user would like to view
    parameters:
    data:(pandas DataFrame)Data frame read from.tsv file containing all required information
    returns:
    choice:returns the choice of year made by the user to the table plotting system
    ratings:returns a ratings dataframe containing the index where the ratings have occurred and the value 
    associated with that index
    max_voted_index:returns the index of the movie that is voted the most(considered as most popular movie)"""
    print("\n"+"-"*10 +"Most popular movies of a year"+ "-"*10)
    year = int( raw_input("Enter the release year: ") );
    
    if year > max( set(data['ReleaseYear']) ) or year < min( set(data['ReleaseYear']) ) :
        print ('The data correponding to this year is not available ! Try again')
        return 
    num_movies = int(raw_input("Enter the number of recommendations you would like to see : "));
    seg_data = data[data["ReleaseYear"]==year];
    ratings = seg_data["numVotes"].nlargest(num_movies);
    temp =  pd.DataFrame(data.loc[list(ratings.index)]);
    temp = temp.sort_values(['averageRating'], ascending=False);
    temp = temp.loc[:, temp.columns != 'tconst']
    display(HTML(temp.to_html()));

def genre_interface(data):
    """Heart of the genre recommender system. Extracts user preferences of genre and the number of movies 
    within the genre that the user would like to view.
    parameters:
    data:(pandas DataFrame)Data frame read from.tsv file containing all required information
    returns:
    choice:returns the choice of genre made by the user to the table plotting system
    ratings:returns a ratings dataframe containing the index where the ratings have occurred 
    and the value associated with that index
    max_voted_index:returns the index of the movie that is voted the most(considered as most popular movie)"""
    print("\n"+"-"*10 + "Most popular movies in a genre" + "-"*10)
    unique_genres = []
    for x1 in set( data['genres'] ):
        unique_genres += x1.split(',') 
    unique_genres = list(set(unique_genres));
    unique_genres.remove('\\n');
    unique_genres.remove('adult');
    s = ", ".join(unique_genres);
    print("List of genres available in the data :\n {}".format(s));
    gen = raw_input("Enter genre of interest :");
    if gen not in unique_genres :
        print ('Incorrect genre input ! Try again')
        return
    index =[];
    for i in data["genres"]:
        index.append(gen in i);
    seg_data = data[index];
    max_voted_index = seg_data["numVotes"].idxmax();
    num_movies = int(raw_input("Enter the number of recommendations you would like to see : "));
    ratings = seg_data["numVotes"].nlargest(num_movies);
    temp =  pd.DataFrame(data.loc[list(ratings.index)]);
    temp = temp.sort_values(['averageRating'], ascending=False);
    temp = temp.loc[:, temp.columns != 'tconst']
    display(HTML(temp.to_html()));
    
def recommender_system(data):
    """Controls the interface for movie recommendation system
    parameters:
    data:(pandas DataFrame)Data frame read from.tsv file containing all required information"""
    print("-"*30 +"Recommender system" + "-"*30)
    print("Please enter how you wish to search for recommendations (search by)\n *name\n *year\n *genre");
    time.sleep(0.2)
    choice = raw_input("Please enter your preference :");
    if choice in ["name","year","genre"]:
        if(choice=="name"):
            name_based_recommender(data);
        elif(choice=="year"):
            year_interface(data)
        elif(choice=="genre"):
            genre_interface(data);
    else:
        print("Incorrect choice ! Try again ...")


def show_plots_movies(movies, choice = 'rating_histogram' ):
    '''Function for ploting a specific information from movies dataset, based on choice
    Input :
    movies : (pandas daatframe) of movies 
    choice : (str) one of the following choices 
        common_movietitles : plot common movie names vs number of movies with those names
        rating_histogram : histogram plot of average rating of movies
        movies_per_year : plot of Number of movies released per year 
        genre_plots : plot the popularity and average rating of movies in a particular genre
        all:  all choices will be plotted'''
    
    choices = ['common_movietitles','rating_histogram','movies_per_year','genre_plots','all']
    assert (choice in choices + ['all'])
    from collections import Counter
    
    if choice == 'common_movietitles' or choice == 'all':
        names = Counter( movies['primaryTitle'] )
        names = dict( names.most_common(n = 15) )
        plt.figure(figsize=(20,5))
        plt.stem(np.arange(15), names.values())
        plt.rc('xtick', labelsize=18); plt.rc('ytick', labelsize=15) 
        plt.xticks(np.arange(15), names.keys() ,rotation=60)
        plt.xlabel('Movie name',fontsize = 15); plt.ylabel('Number of movies',fontsize = 15)
        plt.title('Most common movie titles', fontsize = 17)
        plt.show()
    
    if choice == 'rating_histogram' or choice == 'all':
        plt.figure(figsize=(15,4))
        plt.hist(movies['averageRating'], bins = 20)
        plt.rc('xtick', labelsize=15); plt.rc('ytick', labelsize=15) 
        plt.title('Rating Histogram', fontsize = 15 ); 
        plt.xlabel('Rating', fontsize = 15 ); plt.ylabel('Number of movies', fontsize = 15 )
        plt.show()
    
    if choice == 'movies_per_year' or choice == 'all':
        years = Counter( movies['ReleaseYear'] )
        plt.figure(figsize=(15,5))
        plt.stem(years.keys()[50:-1], years.values()[50:-1])
        plt.rc('xtick', labelsize=13); plt.rc('ytick', labelsize=13) 
        plt.title('Number of movies per year', fontsize = 15 ); 
        plt.xlabel('Year', fontsize = 17 ); plt.ylabel('Number of movies', fontsize = 15 )
        plt.show()
        
    if choice == 'genre_plots' or choice == 'all':
        from matplotlib.pylab import subplots
        temp = ','.join(set(movies['genres']) ).split(',')
        genres = ['family', 'action','adventure', 'fantasy','animation','comedy','sci-fi',
                  'horror','crime','romance','western','thriller','drama','history']
        avrating = []
        avvotes = []
        for i in range(len(genres)):
            index = []
            for j in movies["genres"]:
                index.append( genres[i] in j);
            avrating.append( movies.loc[index,'averageRating'].mean() )
            avvotes.append( movies.loc[index,'numVotes'].mean() )
        
        plt.figure(figsize=(15,4))
        xaxis = list(range(len(genres)))
        plt.stem( xaxis, avrating)
        plt.xticks(xaxis,genres, rotation=40)
        plt.rc('xtick', labelsize=13); plt.rc('ytick', labelsize=13) 
        plt.title('Average rating per genre', fontsize = 17 ); 
        plt.xlabel('Genre', fontsize = 15 ); plt.ylabel('Average rating', fontsize = 15 )
        plt.ylim((5,7))
        plt.show()

        plt.figure(figsize=(15,4))
        xaxis = list(range(len(genres)))
        plt.stem( xaxis, avvotes)
        plt.xticks(xaxis,genres, rotation=40)
        plt.rc('xtick', labelsize=13); plt.rc('ytick', labelsize=13) 
        plt.title('Poppularity per genre', fontsize = 17 ); 
        plt.xlabel('Genre', fontsize = 15 ); plt.ylabel('Number Votes', fontsize = 15 )
        plt.show()

def genre_trend(movies, start = 1970, end = 2017):
    '''function for plotting the trend of a movie genre throughout the years, period defined by start and end
    Input :
    movies : (pandas dataframe) of movie details
    start : (int) start year
    end : (int) end year'''
    print('Available genres : \n')
    print('family, fantasy, sport, biography, crime, romance, animation, music, comedy, war, sci-fi, talk-show, horror, adventure, news, reality-tv, thriller, mystery, short, film-noir, drama, game-show, action, history, documentary, musical, western'  )   
    print('=================')
    temp = movies[ movies['ReleaseYear']>=start ]
    temp = temp[ temp['ReleaseYear']<= end  ]
    genre = raw_input('Enter genre : ').lower()
    index = []
    for i in temp['genres'] : 
        index.append(genre in i)
    grp = temp.groupby('ReleaseYear')
    temp2 = temp[index]
    temp1 = temp2.groupby('ReleaseYear')
    ratings = temp1.mean()['averageRating']
    numvotes = temp1.mean()['numVotes']
    year = set(temp2['ReleaseYear'])
    portion = {i:temp1.get_group(i)['tconst'].count()*1.0/grp.get_group(i)['tconst'].count() for i in year}
    plt.figure(figsize=(16,4) )
    #plt.subplot(221)
    #plt.stem(ratings.index, ratings.values)
    #plt.ylabel('Average rating', fontsize = 14)
    #plt.xlabel('Year', fontsize = 14)
    #plt.title('Average rating of genre, '+genre+' throughout the years', fontsize = 14)
    plt.subplot(121)
    plt.stem(numvotes.index, numvotes.values)
    plt.ylabel('Average number of Votes', fontsize = 14)
    plt.xlabel('Year', fontsize = 14)
    plt.title('Popularity of genre, '+genre+' throughout the years', fontsize = 14)
    plt.subplot(122)
    plt.stem(portion.keys(), portion.values())
    plt.ylabel('Portion', fontsize = 14)
    plt.xlabel('Year', fontsize = 14)
    plt.title('Portion of genre, '+genre+' throughout the years', fontsize = 14)
    plt.savefig('img/'+genre+'.png')
    plt.show()
    
    
def director_trend(movies) :
    '''function to plot the trend of movies associated with a person, throughout the years
    Input :
    movies : (pandas dataframe) movie details'''
    
    directors = ','.join(list( movies.directors ))
    directors = set(directors.split(','))
    while True :
        name = raw_input('Enter director\'s name : ').lower()
        if name not in directors:
            print('Directors with similar name : ')
            dirs = []
            for i in directors : 
                if name in i:
                    dirs.append(i)
            print(dirs)
        if name in directors:
            break
    index = []
    for i in movies['directors'] : 
        index.append(name in i)
    temp = movies[index]
    temp1 = temp.groupby('ReleaseYear')
    ratings = temp1.mean()['averageRating']
    numvotes = temp1.mean()['numVotes']
    plt.figure(figsize=(16,3.5) )
    plt.subplot(121)
    plt.stem(ratings.index, ratings.values)
    plt.ylabel('Average rating', fontsize = 14)
    plt.xlabel('Year', fontsize = 14)
    plt.title('Average rating of '+name+'\'s movies', fontsize = 14)
    plt.ylim([4,10])
    plt.subplot(122)
    plt.stem(numvotes.index, numvotes.values)
    plt.ylabel('Average number of Votes', fontsize = 14)
    plt.xlabel('Year', fontsize = 14)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title('Popularity of '+name+'\'s movies', fontsize = 14)
    plt.show()
