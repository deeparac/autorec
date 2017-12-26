import numpy as np
import pandas as pd
from imdbpie import Imdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    # Only made by Whirldata
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def show_covers(mat, movie_id):
    url = out[out['movieId'] == movie_id]['cover_url'][0]
    img = url_to_image(url)
    plt.imshow(img)

def get_imdb_id(movie_id):
    imdbid = link[link['movieId'] == mid]['imdbId'].values[0]
    pad_num = (len(str(imdbid)))
    paddings = '0' * (7 - pad_num)
    imdbid = 'tt' + paddings + str(imdbid)
    return imdbid

imdb = Imdb(anonymize=True) # to proxy requests

rating_path = './ml-latest-small/ratings.csv'
link_path = './ml-latest-small/links.csv'

rating = pd.read_csv(rating_path)
link = pd.read_csv(link_path)


# for login: temporarily no bird use
ml_base_url = 'https://movielens.org/movies/'

login_url = 'https://movielens.org/login'
login_conf = {
    'inputEmail': 'arac',
    'inputPassword': 'QingWu0apply'
}

# for output
out = rating.copy()

directors = []
casts = []
writers = []
imdb_rating = []
cover_url = []

for index, row in out.iterrows():
    try:
        mid = (row['movieId'].astype(int))
        imdbid = get_imdb_id(mid)
        title = imdb.get_title_by_id(imdbid)
    except:
        print(imdbid)
        break
    idirectors = [x.name for x in title.directors_summary]
    icasts = [x.name for x in title.cast_summary]
    iwriters = [x.name for x in title.writers_summary]
    iimdb_rating = title.rating
    icover_url = title.cover_url
    directors.append(idirectors)
    casts.append(icasts)
    writers.append(iwriters)
    imdb_rating.append(iimdb_rating)
    cover_url.append(icover_url)

out['director'] = directors
out['writer'] = writers
out['casts'] = casts
out['imdb_rating'] = imdb_rating
out['cover_url'] = cover_url

out.to_csv('./augmented.csv')
