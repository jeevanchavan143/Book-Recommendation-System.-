from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
import gzip
from django.conf.urls.static import static
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
import sklearn
from sklearn.decomposition import TruncatedSVD
from django.http import JsonResponse
from book.settings import BASE_DIR
path = BASE_DIR+'/media/'
from multiselectfield import MultiSelectField



def parse(path):
    try:
            g = gzip.open(path, 'rb')
            for l in g:
                yield eval(l)
    except Exception as e:
        print(e)


def getDF(path):
    try:
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
    except Exception as e:
        print(e)        
    return pd.DataFrame.from_dict(df, orient='index')

file = path+ str('ratings_Health_and_Personal_Care.csv')
rating = pd.read_csv(file,encoding='latin-1', header = None, sep = ',',dtype={"0": object, "1": object,"2":np.int64 ,"3":np.int64})
rating.columns = ['UserID','asin','Rating','Timestamp']
total_rating = rating.groupby(by=['asin'])['Rating'].count().reset_index().rename(columns={'Rating':'totalRating'})
rating_totalcount = rating.merge(total_rating,on='asin',how='left')

popularity_threshold = 1000
threshold_matrix = rating_totalcount.query('totalRating>= @popularity_threshold')

rating.groupby(by=['asin'])['Rating'].count().reset_index().rename(columns={'Rating':'totalRating'})

file1 = path+ str('meta_Health_and_Personal_Care.json.gz')
product = getDF(file1)
rated_products = product.iloc[:,0:4][['title','description','asin','imUrl']]
entire_ratedproducts = rated_products.merge(total_rating,on='asin',how='left')

popularity_threshold = 1000
threshold_data_model = entire_ratedproducts.query('totalRating>= @popularity_threshold')
product_threshold_model = threshold_data_model.reset_index(drop=True)
product_threshold_model.iloc[:,1][product_threshold_model.iloc[:,1] == ''] = 'no description available'
product_threshold_model = product_threshold_model.fillna('no description available')

user_rating_threshold_model = threshold_matrix
user_rating_threshold_model["U_ID"]=labelencoder.fit_transform(user_rating_threshold_model.iloc[:,0])
user_rating_threshold_model["P_ID"]=labelencoder.fit_transform(user_rating_threshold_model.iloc[:,1])
user_rating_threshold_model=user_rating_threshold_model[['UserID','asin','Timestamp','totalRating','U_ID','P_ID','Rating']]

input_matrix = user_rating_threshold_model.iloc[:,4:]
# print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',index)input_matrix = user_rating_threshold_model.iloc[:,4:]
input_matrix = input_matrix.pivot_table(columns='U_ID',index='P_ID',values='Rating').fillna(0)





def recommendation(request):

    if request.method=='GET' and'index' in request.GET:
        index = int(request.GET.get('index'));
       
        input_array = csr_matrix(input_matrix)

        from sklearn.neighbors import NearestNeighbors
        model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
        model_knn.fit(input_array)
        pickle_file=str(model)+'.pkl'
        svm_pkl=open(svm_pickle_file,'wb')
        pickle.dump(x1,svm_pkl)
        query_index = index
        distance,indices = model_knn.kneighbors(input_matrix.iloc[query_index,:].values.reshape(1,-1),n_neighbors = 6)
        
        line = '{ "result" : ['
        for count, i in enumerate(indices[0]):
            title = product_threshold_model.iloc[i,0]
            desc = product_threshold_model.iloc[i,1]
            url = product_threshold_model.iloc[i,3]  
            if count == (len(indices[0])-1):
                line += '{ "id":"' + str(i) + '","title":"' + title + '","description":"' + desc + '","url":"' + url + '" }'
            else:
                line += '{ "id":"' +str(i)+ '","title":"' + title + '","description":"' + desc + '","url":"' + url + '" },'
        
        line += '] }'
       # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',line)
        return JsonResponse(line,safe = False)
    return render(request, 'recommendation.html')


def book_rec():
    file2 = path+ str('BX-Books.csv')
    dtype={"0": object, "1": object,"2":object ,"3":object,"4":object,"5":object,"6":object,"7":object}
    book = pd.read_csv(file2, sep=';', error_bad_lines = False, encoding = "latin-1",dtype = dtype)
    book.columns = ['ISBN', 'bookTitle', 'bookAuther', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    
    file3 = path+ str('BX-Users.csv')
    dtype = {'0':np.int64,'1':object,'2':np.float64}
    user = pd.read_csv(file3, sep=';', error_bad_lines = False, encoding = "latin-1",dtype=dtype)
    user.columns = ['userId', 'location', 'age']
    
    file4 = path+ str('BX-Book-Ratings.csv')
    dtype = {'0':np.int64,'1':object,'2':np.int64}
    book_rating = pd.read_csv(file4, sep=';', error_bad_lines = False, encoding = "latin-1",dtype = dtype)
    book_rating.columns = ['userId', 'ISBN', 'bookRating']

    # getting popular book by combining rating data and book data
    combine_book_rating = pd.merge(book_rating, book, on='ISBN')
    columns = ['imageUrlS', 'imageUrlM']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)

    # group by book titles and create a new column for total rating count
    combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])

    book_ratingCount = (combine_book_rating.groupby(by=['bookTitle'])['bookRating']
                        .count().reset_index().rename(columns = {'bookRating' : 'totalRatingCount',})
                    [['bookTitle', 'totalRatingCount']])

    # combine rating data and total rating count data to filter out less known books
    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle', right_on='bookTitle', how='inner')

    popularity_threshold = 50
    rating_popularBook = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold' )


    # In order to improve computing speed, and not run into the “MemoryError” issue, I will limit our user data to 
    # those in the US and Canada. And then combine the user data with rating data and total rating count data.
    combined = rating_popularBook.merge(user, left_on='userId', right_on='userId', how='left')
    us_canada_user_rating = combined[combined['location'].str.contains("usa|canada")]
    us_canada_user_rating = us_canada_user_rating.drop(['age'], axis = 1)

    us_canada_user_rating_pivot2 = us_canada_user_rating.pivot_table(index = 'userId', columns = 'bookTitle', values = 'bookRating').fillna(0)
    X = us_canada_user_rating_pivot2.values.T

    SVD = TruncatedSVD(n_components = 12, random_state = 17)
    matrix = SVD.fit_transform(X)
    corr = np.corrcoef(matrix)
    
    return us_canada_user_rating_pivot2, corr, us_canada_user_rating

#### -----------------------------   ####
us_canada_user_rating_pivot2, corr, us_canada_user_rating = book_rec()





def projectinfo1(request):
    return render(request, 'projectinfo1.html')

def recommendbook(request):
    import json

    if request.method=='GET' and'index' in request.GET:

        title = request.GET.get('index')
       # print('111111111111111111111111111111111111111',title)


        selected_book = []
        recommended_book = []

        us_canada_book_title = us_canada_user_rating_pivot2.columns
        us_canada_book_list = list(us_canada_book_title)
        coffey_hands = us_canada_book_list.index(title)

        corr_coffey_hands = corr[coffey_hands]
        movie_title = list(us_canada_book_title[(corr_coffey_hands < 1.0) & (corr_coffey_hands > 0.8)])

        url = us_canada_user_rating['imageUrlL'][us_canada_user_rating['bookTitle'] == title].head(1).values[0]
        bookAuther = us_canada_user_rating['bookAuther'][us_canada_user_rating['bookTitle'] == title].head(1).values[0]
        publisher = us_canada_user_rating['publisher'][us_canada_user_rating['bookTitle'] == title].head(1).values[0]
        
        selected_book.append(title)
        selected_book.append(url)
        selected_book.append(bookAuther)
        selected_book.append(publisher)
        recommended_book.append(selected_book)

        for i in range(len(movie_title[:6])):
            temp = []
            url = us_canada_user_rating['imageUrlL'][us_canada_user_rating['bookTitle'] == movie_title[i]].head(1).values[0]
            bookAuther = us_canada_user_rating['bookAuther'][us_canada_user_rating['bookTitle'] == movie_title[i]].head(1).values[0]
            publisher = us_canada_user_rating['publisher'][us_canada_user_rating['bookTitle'] == movie_title[i]].head(1).values[0]
            
            temp.append(movie_title[i])
            temp.append(url)
            temp.append(bookAuther)
            temp.append(publisher)
            recommended_book.append(temp)
          
            
            line = '{ "result" : ['
            for count, book in enumerate(recommended_book):
                title = book[0]
                url = book[1]
                author = book[2]
                publisher = book[3]
                print("title - {0} url- {1} author- {2} publisher- {3}".format(title, url, author, publisher))
                if count == (len(recommended_book)-1):
                    line += '{ "id":"' + str(count) + '","title":"' + title + '","author":"' + author + '","publisher":"' + publisher + '","url":"' + url + '" }'
                else:
                    line += '{ "id":"' + str(count) + '","title":"' + title + '","author":"' + author + '","publisher":"' + publisher + '","url":"' + url + '" },'
    
            line += '] }'
       
        return JsonResponse(line,safe = False)
        return render(request, 'recommendbook.html')

        #return render(request, 'recommendbook.html',{'context':context})

    #         cont['title'] = title
    #         cont['url'] = url
    #         cont['author'] = author
    #         cont['publisher'] = publisher
    #         line[f'result_{count}'] = cont
    # print('4444444444444444444444444444',line)
             
    return render(request, 'recommendbook.html')




