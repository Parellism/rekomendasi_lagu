import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import requests
from bs4 import BeautifulSoup

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://e7.pngegg.com/pngimages/921/584/png-clipart-night-sky-star-desktop-night-blue-atmosphere-thumbnail.png");
background-size: cover;
}
</style>
"""

st.set_page_config(
    page_title='Rekomendasi Lagu',
    layout="wide"
)

st.markdown(page_bg_img, unsafe_allow_html=True)

def get_song_genre(lastfm_link):
    response = requests.get(lastfm_link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        genre_list = soup.find(class_='tags-list')
        genres = [genre.text for genre in genre_list.find_all('a')]
        return genres
    else:
        return None

data = pd.read_csv('2017sd2021.csv')
st.title('Rekomendasi Musik')
selected_year = st.slider("Pilih Tahun", 2017, 2021)
data = data[(data['Years'] == selected_year)]
data['genres'] = data[['Genre', 'Genre2', 'Genre3', 'Genre4', 'Genre5', 
                       'Genre6', 'Genre7', 'Genre8', 'Genre9', 
                       'Genre10']].apply(lambda x: ' '.join(x.dropna()), axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['genres'].values.astype('U'))

if selected_year != 0 :
    lastfm_link = st.text_input('Masukkan link Last.fm untuk lagu')

if st.button('Dapatkan Rekomendasi'):

    if lastfm_link:

        genres_link = get_song_genre(lastfm_link)

        if genres_link:

            combined_genres = genres_link + data['genres'].tolist()

            tfidf_matrix_combined = tfidf.fit_transform(combined_genres)

            cosine_sim_combined = linear_kernel(tfidf_matrix_combined, tfidf_matrix_combined)

            index = len(genres_link) 

            similarity_scores = list(enumerate(cosine_sim_combined[index]))

            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            top_similar_songs = similarity_scores[1:6]

            recommended_songs = data.iloc[[i[0] for i in top_similar_songs]][['Track Name', 'Artist']]

            st.subheader('Lagu-lagu yang Direkomendasikan:')
            for i, song in enumerate(recommended_songs.iterrows()):
                st.write(f"{i+1}. {song[1]['Track Name']} - {song[1]['Artist']}")
        else:
            st.error('Gagal mendapatkan genre dari link Last.fm')
    else:
        st.warning('Masukkan link Last.fm untuk lagu terlebih dahulu')
