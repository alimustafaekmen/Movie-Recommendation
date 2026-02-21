# =============================================================================
# Film Öneri Motoru / Movie Recommendation Engine
# =============================================================================

import logging
import datetime
import traceback
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process, fuzz

from src.config import (
    TMDB_API_KEY, TMDB_API_BASE_URL, TMDB_POSTER_BASE_URL, API_TIMEOUT,
    DEFAULT_RECOMMENDATION_COUNT, MAX_RECOMMENDATION_COUNT,
    MIN_VOTES_THRESHOLD, SIMILARITY_WEIGHT, VOTE_AVERAGE_WEIGHT,
    VOTE_COUNT_WEIGHT, POPULARITY_WEIGHT, FUZZY_SCORE_CUTOFF,
    EXCLUDED_GENRES_FOR_POPULATE
)
from src.helpers import parse_json_helper


class RecommendationEngine:
    """
    Film öneri sisteminin ana sınıfı.
    Main class for the movie recommendation system.

    Veri yükleme, benzerlik hesaplama ve öneri üretme işlemleri.
    Handles data loading, similarity calculation, and generating recommendations.
    """

    def __init__(self, movies_path, credits_path):
        """
        Motoru başlat ve verileri yükle.
        Initialize engine and load data.
        """
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies_df = pd.DataFrame()   # İşlenmiş film verisi / Processed movie data
        self.cosine_sim = None             # Benzerlik matrisi / Similarity matrix
        self.poster_cache = {}             # Poster URL önbelleği / Poster URL cache

        # Tür haritaları / Genre maps
        self.genre_id_to_english_name_map = {}
        self.english_genre_name_to_id_map = {}

        # Arayüz listeleri / UI lists
        self.all_genres = []
        self.all_movie_titles = []
        self.all_director_names = []
        self.popular_movies_list = []
        self.popular_directors_list = []

        # Veri yükleme işlemini başlat / Start data loading
        self._load_and_process_data()

    # =========================================================================
    # Veri Yükleme ve İşleme / Data Loading and Processing
    # =========================================================================

    def _load_raw_data(self):
        """
        CSV dosyalarından ham verileri (film ve ekip bilgileri) yükler.
        Loads raw data (movie and credits info) from CSV files.
        """
        logging.info("CSV dosyaları yükleniyor... / Loading CSV files...")
        import os

        # 1. Dosyaların varlığını kontrol et / 1. Check if files exist
        is_movies_file_exists = os.path.exists(self.movies_path)
        is_credits_file_exists = os.path.exists(self.credits_path)

        if not is_movies_file_exists:
            raise FileNotFoundError(f"Film CSV bulunamadı / Movies CSV not found: {self.movies_path}")
        if not is_credits_file_exists:
            raise FileNotFoundError(f"Ekip CSV bulunamadı / Credits CSV not found: {self.credits_path}")

        # 2. Sadece ihtiyacımız olan sütunları oku (Performans için)
        # 2. Read only the columns we need (For performance)
        movie_columns_to_read = [
            'id', 'title', 'overview', 'genres', 'keywords',
            'release_date', 'vote_average', 'vote_count',
            'popularity', 'tagline', 'runtime'
        ]
        credit_columns_to_read = ['movie_id', 'cast', 'crew']

        # Pandas ile dosyaları tabloya çevir / Convert files to tables with Pandas
        movies_df = pd.read_csv(self.movies_path, usecols=movie_columns_to_read)
        credits_df = pd.read_csv(self.credits_path, usecols=credit_columns_to_read)

        # 3. ID (Kimlik) sütunlarını düzelt
        # 3. Fix ID columns
        
        # Hatalı olan harfleri veya boşlukları yoksay (coerce), geriye sayı veya boşluk döner
        # Ignore invalid texts or spaces (coerce), it returns numbers or empty values
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        # ID'si boş olan bozuk filmleri tamamen sil / Completely delete bad movies with empty IDs
        movies_df.dropna(subset=['id'], inplace=True)
        # Kalan geçerli ID'leri tam sayı (integer) yap / Make remaining valid IDs integers
        movies_df['id'] = movies_df['id'].astype(int)

        credits_df['movie_id'] = pd.to_numeric(credits_df['movie_id'], errors='coerce')
        credits_df.dropna(subset=['movie_id'], inplace=True)
        credits_df['movie_id'] = credits_df['movie_id'].astype(int)

        logging.info(f"Veriler yüklendi / Data loaded: Films={movies_df.shape}, Credits={credits_df.shape}")
        return movies_df, credits_df

    def _preprocess_data(self, movies_df, credits_df):
        """
        Ham verileri birleştir ve yazılımın(motorun) kullanabileceği son temiz haline getir.
        Merge raw data and bring it to final clean state that software(engine) can use.
        """
        logging.info("Veri ön işleme başlıyor... / Starting data preprocessing...")

        # 1. İki farklı tabloyu (Filmler ve Ekip) aynı ID üzerinden birleştir
        # 1. Merge two different tables (Movies and Credits) on the matching ID
        df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')

        # 'movie_id' isimli gereksiz kopya sütunu sil
        # Delete redundant copy column named 'movie_id'
        if 'movie_id' in df.columns:
            df = df.drop(columns=['movie_id'])

        # 2. Boş verileri doldur (Programın hata vermemesi için)
        # 2. Fill empty strings (To prevent program from throwing errors)
        fill_values_map = {
            'overview': '', 
            'genres': '[]', 
            'keywords': '[]',
            'cast': '[]', 
            'crew': '[]', 
            'tagline': ''
        }
        df.fillna(fill_values_map, inplace=True)

        # 3. JSON formatında (Metin şeklindeki sözlük) gelen sütunları ayrıştır
        # 3. Parse columns coming in JSON format (Text-based dictionaries)
        
        # Yardımcı Fonksiyonlar / Helper Functions
        # Aşağıda yazılan minik fonksiyonlar, tablodaki her bir sütun için özel olarak çalışacaktır.
        
        def extract_director(crew_json):
            return parse_json_helper(crew_json, job='Director')
            
        def extract_genre_names(genres_json):
            return parse_json_helper(genres_json, key='name')
            
        def extract_keywords(keywords_json):
            return parse_json_helper(keywords_json, key='name')
            
        def extract_top_15_cast(cast_json):
            names_list = parse_json_helper(cast_json, limit=15)
            # Dönen listeyi aralarına virgül koyarak yazıyla dönüştür
            # Convert returned list to text separating by comma
            return ', '.join(names_list)

        # Yardımcı fonksiyonları uygulayarak yeni sütunları oluştur
        # Create new columns by applying helper functions
        df['director'] = df['crew'].apply(extract_director)
        df['genres_str'] = df['genres'].apply(extract_genre_names)
        df['keywords_str'] = df['keywords'].apply(extract_keywords)
        df['cast'] = df['cast'].apply(extract_top_15_cast)

        # 4. Sayısal veri düzenlemeleri
        # 4. Numeric data arrangements
        
        # Çıkış tarihinden sadece YIL bilgisini çek / Extract only YEAR info from release_date
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

        # Hatalı olanları 'coerce' ile sıfır yap (Sistemin çökmesini engeller)
        # Make faulty ones zero with 'coerce' (Prevents system crash)
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0.0)
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).astype(int)
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0.0)
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(0).astype(int)

        # Metin sütunlarını kesin olarak 'str' (metin) tipine dönüştür
        # Strictly convert text columns to 'str' (string) type
        for col in ['title', 'overview', 'director', 'tagline', 'cast']:
            df[col] = df[col].astype(str)

        # 5. Eski işe yaramayan sütunları kaldır ve yenileri adlandır
        # 5. Drop useless old columns and rename new ones
        df = df.drop(columns=['crew', 'release_date', 'genres', 'keywords'], errors='ignore')
        df.rename(columns={'genres_str': 'genres', 'keywords_str': 'keywords'}, inplace=True)

        # 6. ID kontrolünü tekrar yap: Kimliği olmayan filmler ana tabloda yer edinemez
        # 6. Re-check ID: Movies without IDs cannot find place in main table
        if 'id' in df.columns:
            df.dropna(subset=['id'], inplace=True)
            df['id'] = df['id'].astype(int)
        else:
            logging.error("KRİTİK HATA: 'id' sütunu silinmiş veya bulunamadı! / CRITICAL ERROR: 'id' column erased or missing!")
            return pd.DataFrame()

        logging.info(f"Ön işleme başarıyla tamamlandı / Preprocessing successfully done: {df.shape}")
        return df


    def _calculate_similarity(self, df):
        """
        Makine öğrenmesi yöntemiyle (TF-IDF) filmlerin birbirine benzerlik oranını hesaplar.
        Calculates similarity ratio of movies using machine learning (TF-IDF).
        """
        logging.info("Benzerlik matrisi hesaplanıyor... / Calculating similarity matrix...")

        # 1. Filmlerin tüm özelliklerini tek bir uzun metinde (string) birleştiriyoruz
        # 1. We combine all features of movies into a single long string
        combined_text = df['genres'] + ' ' + df['keywords'] + ' ' + df['director'] + ' ' + df['overview'] + ' ' + df['tagline']
        df['combined_features'] = combined_text

        # 2. Metinleri matematiksel sayılara (vektörlere) çeviren aracı hazırlıyoruz
        # İngilizce dolgu kelimeleri ('the', 'and' vb.) yoksayılacak (stop_words='english')
        # 2. Prepare the tool that converts texts to mathematical numbers (vectors)
        # English stop words ('the', 'and' etc.) will be ignored
        vectorizer_tool = TfidfVectorizer(stop_words='english', max_features=20000)
        
        # 3. Metinleri sayılara çeviriyoruz
        # 3. Convert texts to numbers
        number_matrix = vectorizer_tool.fit_transform(df['combined_features'])

        # 4. Kosinüs Benzerliği (Cosine Similarity) formülü ile 
        # her filmin diğer tüm filmlerle benzerlik oranını (0 ile 1 arası) hesaplıyoruz
        # 4. Calculate similarity ratio (between 0 and 1) of each movie with all other movies
        similarity_matrix = cosine_similarity(number_matrix, number_matrix)

        # 5. İşi biten geçici metin birleştirme sütununu siliyoruz
        # 5. Delete the temporary text combination column since we are done with it
        df = df.drop(columns=['combined_features'], errors='ignore')

        logging.info(f"Benzerlik matrisi hesaplandı / Similarity matrix calculated: {similarity_matrix.shape}")
        
        # Orijinal veriyi ve oluşturduğumuz benzerlik tablosunu geri gönderiyoruz
        return df, similarity_matrix

    def _populate_metadata_lists(self):
        """
        Arayüz (Menüler, Dropdown listeler vb.) için tür, film ve yönetmen listelerini oluşturur.
        Builds genre, movie, and director lists for the User Interface (Menus, Dropdowns etc.).
        """
        # Veri yoksa işlemi iptal et / Cancel operation if there is no data
        if self.movies_df.empty:
            return

        logging.info("Arayüz listeleri oluşturuluyor... / Building UI lists...")

        # 1. Tüm eşsiz film türlerini topla / Collect all unique movie genres
        # (Benzersiz olmaları için Python 'set' veri tipi kullanıyoruz / Using 'set' for uniqueness)
        all_genres_set = set()
        
        for genres_string in self.movies_df['genres'].dropna():
            if not genres_string:
                continue
            
            # İçinde birden fazla tür olabilir, boşluklardan böl
            # Might contain multiple genres, split by space
            for genre in genres_string.split():
                clean_genre = genre.strip()
                # Eğer tür geçerliyse ve yasaklı listesinde değilse ekle
                # If genre is valid and not in excluded list, add it
                if clean_genre and clean_genre.lower() not in EXCLUDED_GENRES_FOR_POPULATE:
                    all_genres_set.add(clean_genre)
                    
        # Set'i listeye çevir ve alfabetik sırala / Convert set to list and sort alphabetically
        self.all_genres = sorted(list(all_genres_set))

        # 2. Arama çubuğu için tüm film başlıklarını al ve alfabetik sırala
        # 2. Get all movie titles for search bar and sort alphabetically
        self.all_movie_titles = sorted(self.movies_df['title'].dropna().unique().tolist())

        # 3. Tüm yönetmen isimlerini al ve alfabetik sırala
        # 3. Get all director names and sort alphabetically
        unique_directors = self.movies_df['director'].dropna().unique().tolist()
        self.all_director_names = []
        for name in sorted(unique_directors):
            if name:  # Sadece geçerli, boş olmayan isimleri ekle / Only add valid, non-empty names
                self.all_director_names.append(name)

        # 4. Ana sayfa vitrini için popüler filmleri belirle (Oy Sayısına Göre)
        # 4. Determine popular movies for home showcase (By Vote Count)
        if 'vote_count' in self.movies_df.columns:
            # Sadece yeterli oyu alan filmleri ayrı bir tablo yap
            # Make a separate table for movies with enough votes
            has_enough_votes = self.movies_df['vote_count'] >= MIN_VOTES_THRESHOLD
            highly_voted_movies = self.movies_df[has_enough_votes]
            
            # Oylara göre azalan şekilde (Büyükten küçüğe) sırala
            # Sort descending by votes (Highest to lowest)
            sorted_by_votes = highly_voted_movies.sort_values('vote_count', ascending=False)
            
            # İlk 25 filmin başlığını al listeye çevir
            # Get titles of top 25 movies and convert to list
            self.popular_movies_list = sorted_by_votes['title'].head(25).tolist()
        else:
            # Eğer oylama bilgisi yoksa yedek plan: Rastgele ilk 15 filmi al
            # Fallback plan if no voting info: Take first 15 movies randomly
            self.popular_movies_list = self.all_movie_titles[:15]

        # 5. Ana sayfa vitrini için popüler yönetmenleri belirle (Popülerlik puanına göre)
        # 5. Determine popular directors for home showcase (By Popularity score)
        has_popularity_and_director = ('popularity' in self.movies_df.columns and 'director' in self.movies_df.columns)
        
        if has_popularity_and_director:
            try:
                # Sadece geçerli (Boş olmayan) yönetmeni olan satırları bul
                # Find only rows with valid (Non-empty) directors
                has_valid_director = self.movies_df['director'].notna() & (self.movies_df['director'] != '')
                valid_director_movies = self.movies_df[has_valid_director]
                
                if not valid_director_movies.empty:
                    # Yönetmenlere göre grupla ve popülerlik puanlarının ortalamasını (mean) al
                    # Group by directors and calculate average (mean) of their popularity scores
                    director_avg_popularity = valid_director_movies.groupby('director')['popularity'].mean()
                    
                    # Puanı yüksek olan ilk 20 kişiyi seç
                    # Select top 20 people with high scores
                    top_20_directors = director_avg_popularity.sort_values(ascending=False).head(20)
                    self.popular_directors_list = top_20_directors.index.tolist()
                else:
                    self.popular_directors_list = self.all_director_names[:20]
            except Exception:
                # Bir hata olursa sistem çökmesin, klasik listeyi kullan
                # Don't let system crash on error, use classic list
                self.popular_directors_list = self.all_director_names[:20]
        else:
            self.popular_directors_list = self.all_director_names[:20]

        logging.info(
            f"Listeler oluşturuldu / Lists built: "
            f"{len(self.all_genres)} türler (genres), {len(self.all_movie_titles)} filmler (movies)"
        )

    def _load_genre_map_from_tmdb(self):
        """
        TMDb API'sinden filmlerin tür haritasını (İd -> İngilizce İsim) çeker ve hafızaya yazar.
        Fetches movie genre map (ID -> English Name) from TMDb API and writes to memory.
        """
        logging.info("Tür haritası yükleniyor... / Loading genre map based on API...")
        self.genre_id_to_english_name_map = {}
        self.english_genre_name_to_id_map = {}

        if not TMDB_API_KEY:
            logging.warning("API anahtarı eksik, tür dönüşüm haritası yüklenemedi. / API key missing, map failed.")
            return

        try:
            # 1. API'ye tür listesi için istek adresi oluştur / 1. Create request URL for species list
            api_request_url = f"{TMDB_API_BASE_URL}/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
            
            # 2. İsteği gönder ve yanıtı bekle / 2. Send request and wait for response
            response = requests.get(api_request_url, timeout=API_TIMEOUT)
            
            # Yanıt sorunluysa (Örn 404 Error) programın çökmesini sağla, 'except' bloğuna düşer
            # If response is bad (e.g. 404 Error), raise error to fall into 'except' block
            response.raise_for_status() 
            
            # 3. Gelen JSON formatındaki veriyi Python Sözlüğüne (Dictionary) çevir
            # 3. Convert incoming JSON data to Python Dictionary
            json_data = response.json()
            genres_list = json_data.get('genres', [])

            # 4. Gelen listedeki her türü sistemimizin sözlüğüne kaydet
            # 4. Save each genre in the list to our system's dictionary
            for genre_item in genres_list:
                genre_id = genre_item.get('id')
                genre_name = genre_item.get('name')
                
                if genre_id and genre_name:
                    # Numaradan isme çeviren sözlük (12 -> 'Adventure')
                    # Dictionary converting ID to name (12 -> 'Adventure')
                    self.genre_id_to_english_name_map[genre_id] = genre_name
                    
                    # İsimden numaraya çeviren sözlük ('adventure' -> 12)
                    # Dictionary converting name to ID ('adventure' -> 12)
                    self.english_genre_name_to_id_map[genre_name.lower()] = genre_id

            logging.info(f"{len(self.genre_id_to_english_name_map)} tür başarıyla API'den yüklendi. / genres loaded")
            
        except Exception as error:
            logging.error(f"Tür haritası çekerken iletişim hatası / Genre map connection error: {error}")

    def _load_and_process_data(self):
        """
        Tüm veri yükleme, makine öğrenimi başlatma ve liste oluşturma sürecini sırayla yöneten ana orkestra şefi fonksiyon.
        Main orchestrator function that manages the entire data loading, machine learning start, and list building process in order.
        """
        processing_start_time = datetime.datetime.now()
        logging.info("=" * 20 + " MOTOR YÜKLENİYOR / ENGINE LOADING " + "=" * 20)

        try:
            # ADIM 1: Ham dosyaları yükle / STEP 1: Load raw files
            movies_raw_table, credits_raw_table = self._load_raw_data()

            # ADIM 2: Ham dosyaları temizle ve birleştir / STEP 2: Clean and merge raw files
            self.movies_df = self._preprocess_data(movies_raw_table, credits_raw_table)

            # ADIM 3: Arayüz ve arama ekranları için API'den harita sözlüğü kur 
            # STEP 3: Build map dictionary from API for UI and search screens
            self._load_genre_map_from_tmdb()

            if not self.movies_df.empty:
                # ADIM 4: Temel numara 'id' eksik mi onu son kez doğrula / STEP 4: Verify core 'id' is not missing
                is_id_missing = ('id' not in self.movies_df.columns)
                is_id_not_numeric = not pd.api.types.is_integer_dtype(self.movies_df['id'])
                
                if is_id_missing or is_id_not_numeric:
                    logging.error("KRİTİK: 'id' sütunu sorunlu formatta, yapılamıyor. / CRITICAL: 'id' column has severe issues.")
                    self.cosine_sim = None
                else:
                    # ADIM 5: AI benzerlik motoru için oranları hesapla / STEP 5: Calculate ratios for AI similarity engine
                    self.movies_df, self.cosine_sim = self._calculate_similarity(self.movies_df)

                # ADIM 6: Anasayfa listelerini oluştur / STEP 6: Build homepage lists
                self._populate_metadata_lists()
            else:
                logging.error("Oluşturulan veritabanı tamamen boş, işlemler atlanıyor. / DB is empty, skipping.")
                self.cosine_sim = None

            # ADIM 7: İşlemlerin süresini hesapla / STEP 7: Calculate process duration
            time_elapsed = datetime.datetime.now() - processing_start_time
            logging.info(f"Yükleme işlemleri bitti / Operations finished successfully ({time_elapsed})")

        except FileNotFoundError:
            logging.critical("KRİTİK: Veri dosyaları bulunamadı / CRITICAL: Data files not found")
            self.movies_df = pd.DataFrame()
            self.cosine_sim = None
        except Exception as e:
            logging.critical(f"Veri yükleme hatası / Data loading error: {e}\n{traceback.format_exc()}")
            self.movies_df = pd.DataFrame()
            self.cosine_sim = None

    # =========================================================================
    # Film Poster ve Video / Movie Poster and Video
    # =========================================================================

    def get_movie_poster_url(self, movie_id, size='w342'):
        """
        TMDb API'sinden filmin afiş (poster) adresini alır.
        Gets movie poster URL from TMDb API.
        Aynı görseli defalarca indirmemek için önbellek (cache) sistemini kullanır.
        Uses cache system to avoid downloading the same image repeatedly.
        """
        if not TMDB_API_KEY:
            return None

        # 1. Girilen ID değerini sayısal formata sığdırmaya çalışır
        # 1. Tries to fit entered ID value into numeric format
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            # Sayılamaz bir format gelirse (None veya 'abc' gibi), iptal et
            # If an uncountable format comes (like None or 'abc'), cancel
            return None

        # 2. Önbellek (Cache) Kontrolü / 2. Cache Check
        # O filme ve boyuta ait poster daha önce çekilmiş mi bak
        # Check if poster for that movie and size was fetched before
        cache_key_string = f"{movie_id}_{size}"
        
        if cache_key_string in self.poster_cache:
            # Varsa doğrudan sistemdeki hafızadan (sözlükten) gönder, API'yi meşgul etme
            # If exists send directly from system memory (dictionary), don't busy the API
            return self.poster_cache[cache_key_string]

        # 3. Önbellekte yoksa API'ye internet üzerinden istek at
        # 3. If not in cache, make internet request to API
        try:
            api_request_address = f"{TMDB_API_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&language=tr-TR"
            response_data = requests.get(api_request_address, timeout=API_TIMEOUT)
            
            # Bağlantıda hata olduysa (404 Bulunamadı vb.) çökmesi için zorla / Force crash if connection error (404 Not Found etc.)
            response_data.raise_for_status()

            # JSON paketinin içinden sadece poster yolunu ('poster_path') çıkar
            # Extract only poster path ('poster_path') from JSON package
            api_poster_path = response_data.json().get('poster_path')
            
            if api_poster_path:
                # 4. Yarım gelen adresi tam uzun web adresine dönüştür
                # 4. Convert incoming half-address to full long web address
                full_web_url = f"{TMDB_POSTER_BASE_URL}{size}{api_poster_path}"
                
                # Bir sonraki sefere hızlı vermek için sözlüğe (Cache) not al
                # Take note in dictionary (Cache) to give it fast next time
                self.poster_cache[cache_key_string] = full_web_url
                return full_web_url
            else:
                # İnternette poster yoksa 'Boş (None)' cevabını kaydet ve geri dön
                # If poster doesn't exist on internet, save 'Empty (None)' and return
                self.poster_cache[cache_key_string] = None
                return None
                
        except Exception:
            # Ağ sorunu veya başka bir sistem hatası çıkarsa boş yanıt oluştur
            # If network issue or other system error occurs, create empty response
            self.poster_cache[cache_key_string] = None
            return None

    def get_movie_videos(self, movie_id):
        """
        TMDb API'sinden YouTube fragman kimliğini (key) alır.
        Gets YouTube trailer ID (key) from TMDb API.
        """
        if not TMDB_API_KEY:
            return None

        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            return None

        try:
            # 1. API'ye sadece fragman bilgileri ('videos') için istek atıyoruz
            # 1. Sending request to API only for trailer info ('videos')
            api_request_address = f"{TMDB_API_BASE_URL}/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=tr-TR"
            response_data = requests.get(api_request_address, timeout=API_TIMEOUT)
            response_data.raise_for_status()
            
            # Veriyi sözlük olarak al / Get data as dictionary
            json_response = response_data.json()
            videos_list = json_response.get('results', [])

            # 2. Öncelik sırası oluşturuyoruz: Önce Resmi Fragman, sonra Resmi Teaser...
            # 2. Creating priority list: Official Trailer first, then Official Teaser...
            video_types_to_look_for = ['Trailer', 'Teaser']
            
            for desired_type in video_types_to_look_for:
                # SÜZGEÇ 1: Sadece RESMİ videoları bul
                # FILTER 1: Find only OFFICIAL videos
                for video_info in videos_list:
                    # Video YouTube'da mı? Doğru türde mi? Resmi mi?
                    # Is video on YouTube? Is it correct type? Is it official?
                    is_on_youtube = (video_info.get('site') == 'YouTube')
                    is_correct_type = (video_info.get('type') == desired_type)
                    is_brand_official = video_info.get('official')
                    
                    if is_on_youtube and is_correct_type and is_brand_official:
                        return video_info.get('key')
                        
                # SÜZGEÇ 2: Resmi bulamadıysak en azından aynı tipteki gayri-resmi videoları ver
                # FILTER 2: If we couldn't find official, at least give unofficial videos of same type
                for video_info in videos_list:
                    is_on_youtube = (video_info.get('site') == 'YouTube')
                    is_correct_type = (video_info.get('type') == desired_type)
                    
                    if is_on_youtube and is_correct_type:
                        return video_info.get('key')

            # Hiçbir şey eşleşmediyse boş dön
            # If nothing matched return empty
            return None
        except Exception:
            return None

    # =========================================================================
    # Tür İşleme / Genre Processing
    # =========================================================================

    def _process_genres_for_movie_data(self, api_genres_list, movie_title=""):
        """
        API'den gelen karmaşık tür (genre) listesini, sistemimizin 
        arayüzde (HTML) gösterebileceği daha basit sözlüklere (dictionary) çevirir.
        
        Processes complex genre list from API and converts it into 
        simpler dictionaries that our system can show on UI (HTML).

        Returns:
            tuple: (Liste: Arayüz için tür detayları, Metin: İngilizce tür isimleri)
            tuple: (List: Genre details for UI, String: English genre names)
        """
        formatted_genre_details = []
        unique_english_names = set() # Küme (set) kullanıyoruz ki aynı işlem iki kere eklenmesin / Using set to avoid duplicates

        # 1. Liste boşsa işlemi iptal et ve boş değerler döndür
        # 1. Cancel process and return empty values if list is empty
        is_list_empty = not api_genres_list
        if is_list_empty:
            return [], ""

        # 2. Listedeki her bir tür nesnesini (sözlüğünü) tek tek incele
        # 2. Examine each genre object (dictionary) in the list one by one
        for genre_obj in api_genres_list:
            # Tür numarası (ID) ve Türkçe ismi (name) çek / Extract genre ID and Turkish name
            genre_id = genre_obj.get('id')
            display_name = genre_obj.get('name')

            # Herhangi biri eksikse bu türü atla / Skip this genre if any is missing
            is_invalid_genre = not genre_id or not display_name
            if is_invalid_genre:
                continue

            # 3. Kendi oluşturduğumuz haritadan (sözlükten) İngilizce eşdeğerini bul
            # 3. Find the English equivalent from our mapped dictionary
            english_name = self.genre_id_to_english_name_map.get(genre_id)
            
            # Eğer çeviri başarılıysa listelere ekle
            # If translation is successful, add to lists
            if english_name:
                # HTML tarafında kutucuklar (badge) için kullanılacak sözlüğü hazırla
                # Prepare the dictionary to be used for badges on the HTML side
                formatted_genre_details.append({
                    'id': genre_id,
                    'display_name': display_name,   # Ekranda "Aksiyon" vb. yazacak / Will show "Aksiyon" etc. on screen
                    'link_name': english_name,      # Filtreleme için "Action" arayacak / Will search "Action" for filtering
                })
                
                # Eşsiz isimler havuzuna ekle
                # Add to unique names pool
                unique_english_names.add(english_name)

        # 4. Kümedeki tüm İngilizce isimleri A'dan Z'ye sırala ve aralarına boşluk koyarak tek bir yazı (string) yap
        # 4. Sort all English names in set from A to Z and join them with space to make a single string
        sorted_names_list = sorted(list(unique_english_names))
        final_english_string = ' '.join(sorted_names_list)

        return formatted_genre_details, final_english_string

    # =========================================================================
    # Sonuç Formatlama / Result Formatting
    # =========================================================================

    def _format_results(self, df_recommendations, num_to_return):
        """
        Önerilen filmlerin veritabanı (Pandas DataFrame) formatını, 
        web arayüzünün (HTML) anlayabileceği "Sözlükler Listesi" formatına çevirir.
        
        Converts recommended movies database (Pandas DataFrame) format to 
        "List of Dictionaries" format which web UI (HTML) can understand.
        """
        formatted_results_list = []

        # Eğer data boşsa formata sokulacak şey yoktur
        # If data is empty there is nothing to format
        is_dataframe_empty = df_recommendations.empty
        has_id_column = ('id' in df_recommendations.columns)
        
        if is_dataframe_empty or not has_id_column:
            return formatted_results_list

        # 1. Sadece kullanıcının istediği sayı ('num_to_return') kadar filmi al
        # 1. Only get movies up to the number ('num_to_return') user requested
        df_limited = df_recommendations.head(num_to_return).copy()

        # 2. Sınırlanmış film listesinde her bir filmin üzerinden teker teker satır satır geç
        # 2. Iterate row by row over each movie in the limited movie list
        for _, row_data in df_limited.iterrows():
            movie_item_dict = row_data.to_dict()

            # 3. Film Numarasını (ID) çek ve doğrula / 3. Fetch and validate Movie ID
            raw_id_value = movie_item_dict.get('id')
            
            # ID değeri boşşsa (Not a Number - isna), bu filmi hesaplamaya katma
            # If ID value is empty (Not a Number - isna), skip this movie
            if pd.isna(raw_id_value):
                continue
                
            try:
                valid_movie_id = int(raw_id_value)
            except (ValueError, TypeError):
                continue

            # 4. Yıl verisini temizle / 4. Clean year data
            raw_year = movie_item_dict.get('year', 0)
            try:
                clean_year = int(float(raw_year)) if pd.notna(raw_year) else 0
            except (ValueError, TypeError):
                clean_year = 0

            # 5. Puan Ortalama verisini yuvarla (Örn: 7.8242 -> 7.8)
            # 5. Round Vote Average data (E.g: 7.8242 -> 7.8)
            raw_vote_avg = movie_item_dict.get('vote_average', 0.0)
            try:
                clean_vote_avg = round(float(raw_vote_avg), 1) if pd.notna(raw_vote_avg) else 0.0
            except (ValueError, TypeError):
                clean_vote_avg = 0.0

            # 6. Oyuncu Kadrosunun Sadece İlk 3'ünü HTML için hazırla
            # 6. Prepare Only First 3 of Actor Cast for HTML
            full_cast_text = str(movie_item_dict.get('cast', ''))
            
            # İsimleri virgüle göre ayrı listelere ayır (Örn: "Brad, Tom" -> ["Brad", "Tom"])
            # Split names into separate lists by comma
            all_actor_names = full_cast_text.split(',')
            
            clean_actor_list = []
            for name in all_actor_names:
                trimmed_name = name.strip() # Boşlukları sil / Trim spaces
                if trimmed_name:
                    clean_actor_list.append(trimmed_name)
                    
            # Listenin ilk 3 elemanını al ve aralarına virgül + boşluk koyarak yazdır
            # Take first 3 elements of list and print putting comma + space between
            display_cast = ', '.join(clean_actor_list[:3])
            
            # Eğer 3 kişiden daha fazla oyuncu varsa "ve dahası (...)" işareti koy
            # If there are more than 3 actors put "and more (...)" sign
            if len(clean_actor_list) > 3:
                display_cast += ", ..."

            # 7. Sonuç Sözlüğüne Ekleyip Listeye Gönder 
            # 7. Add to Result Dictionary and Send to List
            formatted_results_list.append({
                'id': valid_movie_id,
                'poster_url': self.get_movie_poster_url(valid_movie_id, size='w342'),
                'year': clean_year,
                'vote_average': clean_vote_avg,
                'title': str(movie_item_dict.get('title', 'N/A')),
                'cast': display_cast,
                'director': str(movie_item_dict.get('director', '')),
            })

        return formatted_results_list

    # =========================================================================
    # Film Arama / Movie Search
    # =========================================================================

    def _find_best_match(self, query, choices, scorer=fuzz.WRatio, score_cutoff=FUZZY_SCORE_CUTOFF):
        """
        Arama metnine en çok benzeyen ('bulanık eşleştirme' / 'fuzzy matching') metni bulur.
        Finds the most similar ('fuzzy matching') text to the search query.
        """
        # Arama kelimesi veya liste boşsa iptal et / Cancel if query or list is empty
        if not query or not choices:
            return None

        try:
            # Aramayı küçük harfe çevir / Convert query to lowercase
            query_lower = query.lower()

            # Tüm seçenekleri de küçük harfe çevirerek yeni bir liste oluştur
            # Create a new list converting all choices to lowercase
            choices_lower = []
            for choice in choices:
                if choice is not None:
                    lower_text = str(choice).lower()
                    choices_lower.append(lower_text)

            # Geçerli seçenek kalmadıysa iptal et / Cancel if no valid choices left
            if len(choices_lower) == 0:
                return None

            # Sistem en iyi eşleşen metni (result) ve benzerlik puanını bulur
            # The system finds the best matching text (result) and its similarity score
            match_result = process.extractOne(
                query_lower, 
                choices_lower, 
                scorer=scorer, 
                score_cutoff=score_cutoff
            )

            # Eğer bir eşleşme puanı sınırı geçtiyse (başarılıysa)
            # If a match passed the score cutoff (successful)
            if match_result is not None:
                # Eşleşen kelimenin küçük harfli halini al
                # Get the lowercase version of the matched word
                best_match_lower = match_result[0]

                # Orijinal (büyük/küçük harf içeren) halini bulmak için ilk listeyi kontrol et
                # Check the first list to find the original (with uppercase/lowercase) version
                for index in range(len(choices)):
                    original_choice = choices[index]
                    if original_choice is not None:
                        if str(original_choice).lower() == best_match_lower:
                            # Orijinal kelimeyi bulduk, geri döndür
                            # Found the original word, return it
                            return original_choice

                # Orijinalini bulamazsak küçük harfli halini mecburen döndür
                # If couldn't find original, return lowercase version
                return best_match_lower

            # Eşleşme bulunamadıysa None döndür / Return None if no match found
            return None
            
        except Exception as error:
            # Hata oluşursa programı çökertme, boş dön
            # Don't crash program on error, return empty
            logging.error(f"Eşleştirme sırasında hata / Matching error: {error}")
            return None

    # =========================================================================
    # Hibrit Skor Hesaplama / Hybrid Score Calculation
    # =========================================================================

    def _calculate_hybrid_score(self, df_candidates, similarity_scores_map):
        """
        Aday filmler için birden fazla durumu birleştirerek (hibrit) bir kalite puanı hesaplar.
        Calculates a combined (hybrid) quality score for candidate movies.

        Formül / Formula: 
        Hibrit Puan = (Benzerlik * 0.50) + (Oy Ortalaması * 0.20) + (Oy Sayısı * 0.10) + (Popülerlik * 0.20)
        """
        # Hesaplama yapılacak film yoksa işlemi iptal et
        # Cancel if there are no movies to calculate
        if df_candidates.empty:
            return df_candidates.copy()

        # Orijinal veriyi bozmamak için kopyasını al
        # Copy to avoid modifying original data
        df = df_candidates.copy()

        # Eksik sütunları veya hatalı veri tiplerini 0 ile doldur
        # Fill missing columns or wrong data types with 0
        for col in ['vote_count', 'popularity', 'vote_average']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                if not is_numeric:
                    df[col] = 0.0

        # Tüm sayıları 0 ile 1 arasına sıkıştıracağız (Normalleştirme / Normalization)
        # We will squeeze all numbers between 0 and 1 (Normalization)
        
        # En çok oy alan filmin oyunu ve en popüler filmin puanını bul (Minimum 1 kabul et)
        # Find max votes and popularity (Accept min 1 to avoid dividing by zero)
        max_votes = max(df['vote_count'].max(), 1.0)
        max_popularity = max(df['popularity'].max(), 1.0)

        # Her filmin değerini en yüksek değere bölersek tüm sayılar 0-1 arasına girer
        # If we divide each movie's value by the highest value, all numbers fall between 0-1
        df['normalized_vote_count'] = df['vote_count'].fillna(0) / max_votes
        df['normalized_popularity'] = df['popularity'].fillna(0) / max_popularity
        
        # Oylar zaten 10 üzerinden olduğu için 10'a bölüyoruz
        # Votes are already out of 10, so we divide by 10
        df['normalized_vote_average'] = df['vote_average'].fillna(0) / 10.0

        # Dışarıdan gönderilen 'similarity_scores_map' yardımıyla her filme benzerlik puanını (0-1 arası) ekle
        # Add similarity score (0-1) to each movie using the provided 'similarity_scores_map'
        def get_similarity(index):
            # Sözlükte bulamazsa 0 döndür / Return 0 if not found in dictionary
            return similarity_scores_map.get(index, 0.0)
            
        df['similarity_score'] = df.index.map(get_similarity).fillna(0.0)

        # Nihai puanı (Hibrit Skoru) ağırlık formülüyle hesapla
        # Calculate the final score (Hybrid Score) using the weight formula
        df['hybrid_score'] = (
            (df['similarity_score'] * SIMILARITY_WEIGHT) +
            (df['normalized_vote_average'] * VOTE_AVERAGE_WEIGHT) +
            (df['normalized_vote_count'] * VOTE_COUNT_WEIGHT) +
            (df['normalized_popularity'] * POPULARITY_WEIGHT)
        )

        return df

    # =========================================================================
    # Benzerlik ile Öneri / Recommendation by Similarity
    # =========================================================================

    def recommend_by_similarity(self, movie_title_query, num=DEFAULT_RECOMMENDATION_COUNT, exclude_movie_id=None):
        """
        Kullanıcının seçtiği bir filme benzeyen diğer filmleri bulur ve önerir.
        Finds and recommends movies similar to a user-selected movie.

        Returns:
            tuple: (recommendations_list, warnings_list)
        """
        warnings = []

        # 1. Giriş Kontrolleri / Input Checks
        if not movie_title_query:
            return [], ["Lütfen aramak için bir film ismi girin veya listeden seçin."]
            
        if self.movies_df.empty or self.cosine_sim is None:
            return [], ["Öneri sistemi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."]

        query = movie_title_query.strip()

        # 2. Aranan Filmi Veritabanında Bulma / Find the Searched Movie in Database
        target_index = None
        matched_title = None

        # Önce birebir, tam eşleşme var mı diye bak (Büyük/Küçük harf duyarsız)
        # First check for an exact match (Case insensitive)
        exact_matches = self.movies_df[self.movies_df['title'].str.lower() == query.lower()]
        
        if not exact_matches.empty:
            # Tam eşleşme bulunduysa ilk sıradakini al
            # If exact match found, take the first one
            matched_title = exact_matches.iloc[0]['title']
            target_index = exact_matches.index[0]
        else:
            # Tam eşleşme yoksa, kelimeye en çok benzeyen (fuzzy match) ismi bul
            # If no exact match, find the most similar sounding name
            matched_title = self._find_best_match(query, self.all_movie_titles)
            
            if matched_title:
                warnings.append(f"'{query}' adında tam bir eşleşme bulunamadı. Bunun yerine en yakın isim olan '{matched_title}' için sonuçlar gösteriliyor.")
                
                # Benzer bulunan ismin tablodaki yerini (index) bul
                # Find the index of the fuzzy matched name in the table
                fuzzy_matches = self.movies_df[self.movies_df['title'] == matched_title]
                if not fuzzy_matches.empty:
                    target_index = fuzzy_matches.index[0]
                else:
                    warnings.insert(0, f"'{matched_title}' isimli film veritabanında kayıp.")
                    return [], warnings
            else:
                warnings.insert(0, f"'{query}' ismine benzer bir film bulunamadı.")
                return [], warnings

        # Matematiksel sıranın (index) doğruluğunu son bir kez teyit et
        # Verify the mathematical index validity one last time
        if target_index is None or target_index not in self.movies_df.index:
            warnings.insert(0, f"'{matched_title}' isimli filmin verilerinde bir sorun var.")
            return [], warnings

        # 3. Benzer Filmleri Hesaplama / Calculating Similar Movies
        try:
            # Matristeki yerini kontrol et / Check its position in the matrix
            if target_index >= self.cosine_sim.shape[0]:
                warnings.insert(0, f"'{matched_title}' için benzerlik bilgisi henüz oluşturulmamış.")
                return [], warnings

            # Bu filmin diğer tüm filmlerle olan benzerlik yüzdelerini al
            # Get similarity percentages of this movie with all other movies
            similarity_percentages = self.cosine_sim[target_index]

            scored_movies = []
            
            # Tüm filmlerin benzerlik puanlarını tek tek döngüye sok
            # Loop through similarity scores of all movies one by one
            for index_number, score in enumerate(similarity_percentages):
                # Sistemdeki film sayısını aşma (Güvenlik kontrolü)
                # Don't exceed total movie count (Security check)
                if index_number >= len(self.movies_df):
                    continue
                    
                # Film kendisi ise listeye alma (Kendisine %100 benzer zaten)
                # Don't add the movie itself to the recommendations
                if index_number == target_index:
                    continue
                    
                # Eğer hariç tutulması istenen bir film varsa onu da alma
                # Skip if there's a specific movie we specifically want to exclude
                if exclude_movie_id is not None:
                    current_movie_id = self.movies_df.iloc[index_number]['id']
                    if current_movie_id == exclude_movie_id:
                        continue
                        
                # Uygun filmi ve puanını listeye ekle
                # Add valid movie and its score to the list
                scored_movies.append((index_number, score))

            # Filmleri en yüksek benzerlik puandan en düşüğe doğru (Büyükten -> Küçüğe) sırala
            # Sort movies from highest similarity score to lowest (Descending)
            scored_movies.sort(key=lambda item: item[1], reverse=True)

            # Sadece ihtiyacımız olan miktarın 3 katı kadar (havuz) film alıyoruz (daha hızlı işlem için)
            # We only take 3 times the requested amount as a candidate pool (for faster processing)
            pool_size = (num * 3) + 1
            candidates = scored_movies[:pool_size]

            if len(candidates) == 0:
                warnings.insert(0, f"'{matched_title}' için önerilebilecek hiçbir benzer film bulunamadı.")
                return [], warnings

            # 4. Adayları Gelişmiş Puanlama (Hibrit Skor) İle Değerlendirme / Evaluating Candidates with Hybrid Score
            
            # Aday filmlerin index numaralarını ayır / Separate candidate index numbers
            valid_indices = []
            similarity_dictionary = {}
            
            for index_num, score in candidates:
                if index_num < len(self.movies_df):
                    valid_indices.append(index_num)
                    similarity_dictionary[index_num] = score

            if len(valid_indices) == 0:
                warnings.insert(0, f"'{matched_title}' için geçerli olabilecek aday çıkmadı.")
                return [], warnings

            # Adayların bilgilerini (satırlarını) yeni bir tablo olarak al
            # Get candidate information (rows) as a new table
            df_candidates = self.movies_df.iloc[valid_indices].copy()

            # Yeni yazdığımız _calculate_hybrid_score fonksiyonu ile kompleks kalite puanı hesapla
            # Calculate complex quality score with our newly written _calculate_hybrid_score function
            df_candidates = self._calculate_hybrid_score(df_candidates, similarity_dictionary)

            # 5. Sonuçları Sıralama ve Filtreleme / Sorting and Filtering Results
            
            # Sadece belli bir oy barajını geçen (emin olduğumuz) filmleri ayır
            # Separate movies that passed a certain vote threshold
            df_highly_voted = df_candidates[df_candidates['vote_count'] >= MIN_VOTES_THRESHOLD]

            if not df_highly_voted.empty:
                # Güvenilir filmler varsa, onları hibrit puana göre sırala
                # If there are reliable movies, sort them by hybrid score
                final_table = df_highly_voted.sort_values('hybrid_score', ascending=False)
            elif not df_candidates.empty:
                # Eğer güvenilir film yoksa mecburen olanları sırala
                # If no reliable movies, sort the existing ones
                final_table = df_candidates.sort_values('hybrid_score', ascending=False)
                warnings.append(f"'{matched_title}' için çok fazla oy almış popüler benzer film bulamadık. Daha az bilinen filmleri listeliyoruz.")
            else:
                warnings.insert(0, f"'{matched_title}' adlı filme benzeyen film listesi boş.")
                return [], warnings

            # 6. Sonuçları Görüntülenecek Formata Çevirme / Formatting Results for Display
            # _format_results fonksiyonu dataları HTML şablonuna uydurur
            final_formatted_list = self._format_results(final_table, num)
            return final_formatted_list, warnings

        except Exception as error:
            # Beklenmedik durumlarda kodun çökmesini engelle
            # Prevent code crash on unexpected situations
            logging.error(f"Benzerlik önerisinde kritik hata / Critical error in similarity recommendation: {error}\n{traceback.format_exc()}")
            warnings.insert(0, "Benzer filmleri ararken beklenmeyen teknik bir sorun oluştu.")
            return [], warnings

    # =========================================================================
    # Özelliklere Göre Öneri / Recommendation by Features
    # =========================================================================

    def recommend_by_features(self, genres=None, director_query=None,
                              min_rating=None, min_year=None, max_year=None,
                              num=DEFAULT_RECOMMENDATION_COUNT):
        """
        Kullanıcının seçtiği kriterlere (tür, yönetmen, puan, yıl) uyan filmleri bulur ve önerir.
        Recommends movies based on user-selected criteria (genre, director, rating, year).
        """
        # 1. Giriş Kontrolleri / Input Checks
        if self.movies_df.empty:
            return [], ["Öneri sistemi şu anda kullanılamıyor."]

        # Orijinal veriyi bozmamak için kopyasını al / Copy data to avoid modifying original
        df = self.movies_df.copy()
        warnings = []
        filter_count = 0 # Kaç tane filtre seçildiğini sayacağız / We'll count selected filters

        # ---------------------------------------------------------
        # 2. Filtre Hazırlıkları / Preparing Filters
        # ---------------------------------------------------------

        # A) Tür Filtresi Hazırlığı / Genre Filter Prep
        selected_genres = []
        if genres:
            for genre in genres:
                if genre:
                    clean_genre = genre.strip().lower()
                    selected_genres.append(clean_genre)
                    
        if len(selected_genres) > 0:
            filter_count += 1

        # B) Yönetmen Filtresi Hazırlığı / Director Filter Prep
        director_match = None
        if director_query:
            director = director_query.strip()
            
            if director:
                filter_count += 1
                
                # Önce tam eşleşme ara / First, look for exact match
                exact_matches = df[df['director'].str.lower() == director.lower()]
                if not exact_matches.empty:
                    director_match = exact_matches.iloc[0]['director']
                else:
                    # Tam eşleşme yoksa bulanık arama (fuzzy search) yap
                    # If no exact match, do fuzzy search
                    director_match = self._find_best_match(director, self.all_director_names)
                    if director_match:
                        warnings.append(f"Yönetmen '{director}' bulunamadı. '{director_match}' kullanılıyor.")
                    else:
                        warnings.append(f"'{director}' adlı yönetmen bulunamadı.")

        # C) Puan ve Yıl Filtreleri Hazırlığı / Rating and Year Filters Prep
        try:
            # Metin olarak gelen sayıları gerçek matematiksel sayılara çevir (float ve int)
            # Convert incoming text numbers to real mathematical numbers (float and int)
            min_rating_f = None
            if min_rating and str(min_rating).strip():
                min_rating_f = float(min_rating)
                
            min_year_f = None
            if min_year and str(min_year).strip():
                min_year_f = int(min_year)
                
            max_year_f = None
            if max_year and str(max_year).strip():
                max_year_f = int(max_year)
                
        except ValueError:
            warnings.append("Geçersiz puan veya yıl formatı girildi. Lütfen sadece sayı kullanın.")
            min_rating_f = None
            min_year_f = None
            max_year_f = None

        if min_rating_f is not None:
            filter_count += 1
        if min_year_f is not None:
            filter_count += 1
        if max_year_f is not None:
            filter_count += 1

        # Mantık kontrolü: Başlangıç yılı bitiş yılından büyük olamaz
        # Logic check: Start year cannot be greater than end year
        if min_year_f is not None and max_year_f is not None:
            if min_year_f > max_year_f:
                warnings.append(f"Başlangıç yılı ({min_year_f}) bitiş yılından ({max_year_f}) büyük olamaz.")
                # Hatalıysa bu filtreleri iptal et / If invalid, cancel these filters
                min_year_f = None
                max_year_f = None

        # Eğer hiç geçerli filtre girilmemişse uyar
        # Warn if no valid filters were entered
        if filter_count == 0:
            return [], ["Lütfen arama yapmak için en az bir kriter (tür, yönetmen vb.) girin."] + warnings

        # ---------------------------------------------------------
        # 3. Tablodaki Verileri Filtreleme / Filtering Table Data
        # ---------------------------------------------------------
        try:
            # Tüm türleri içeren filmleri seç / Select movies containing all requested genres
            if len(selected_genres) > 0:
                for genre in selected_genres:
                    # Tabloda 'genres' sütununda bu tür adının geçtiği satırları (filmleri) tut
                    # Keep rows (movies) where this genre name appears in 'genres' column
                    df = df[df['genres'].str.lower().str.contains(genre, na=False, regex=False)]

            # Seçili yönetmenin filmlerini seç / Select movies by the chosen director
            if director_match:
                df = df[df['director'] == director_match]

            # Minimum puanı geçen filmleri seç / Select movies passing minimum rating
            if min_rating_f is not None:
                df = df[df['vote_average'] >= min_rating_f]

            # Belirtilen yıllar arasındaki filmleri seç (Yılı 0 olmayan geçerli veriler)
            # Select movies between specified years (Valid data where year is not 0)
            if min_year_f is not None:
                is_valid_year = df['year'] != 0
                is_after_min_year = df['year'] >= min_year_f
                df = df[is_after_min_year & is_valid_year]

            if max_year_f is not None:
                is_valid_year = df['year'] != 0
                is_before_max_year = df['year'] <= max_year_f
                df = df[is_before_max_year & is_valid_year]

        except Exception as error:
            logging.error(f"Filtreleme hatası / Filtering error: {error}")
            return [], ["Filtreleme işlemi sırasında teknik bir hata oluştu."] + warnings

        # Filtreleme bittiğinde geriye hiç film kalmamışsa iptal et
        # Cancel if no movies left after filtering
        if df.empty:
            return [], ["Seçtiğiniz kriterlere tam olarak uyan hiçbir film bulunamadı."] + warnings

        # ---------------------------------------------------------
        # 4. Kalan Filmleri Puanlama ve Sıralama / Scoring and Sorting Remaining Movies
        # ---------------------------------------------------------

        # Yalnızca yeterli oy sayısına (güvenilirliğe) sahip filmleri dikkate al
        # Only consider movies with enough votes (reliability)
        df_reliable = df[df['vote_count'] >= MIN_VOTES_THRESHOLD].copy()

        if df_reliable.empty:
            warnings.append(f"Kriterlere uyan {len(df)} film bulundu ancak hiçbirinin yeterli oy sayısı (güvenilirliği) yok.")
            return [], warnings

        # Filmin kalitesini belirlemek için Özel Puan (Feature Score) hesapla
        # Calculate a Feature Score to determine movie quality
        
        # Oyların ve popülerliğin maksimum değerlerini bul (0'a bölme hatasını engellemek için min 1 al)
        # Find max values for votes and popularity (take min 1 to avoid divide by zero errors)
        max_vote_count = max(df_reliable['vote_count'].max(), 1.0)
        max_popularity = max(df_reliable['popularity'].max(), 1.0)

        # Kalite Formülü: %40 Puan ortalaması + %45 Popülerlik + %15 Oy sayısı
        # Quality Formula: 40% Rating average + 45% Popularity + 15% Vote count
        df_reliable['feature_score'] = (
            (df_reliable['vote_average'].fillna(0) / 10.0) * 0.40 +
            (df_reliable['popularity'].fillna(0) / max_popularity) * 0.45 +
            (df_reliable['vote_count'].fillna(0) / max_vote_count) * 0.15
        )

        # Filmleri en yüksek puandan en düşüğe doğru sırala
        # Sort movies from highest score to lowest
        final_sorted_table = df_reliable.sort_values('feature_score', ascending=False)
        
        # _format_results fonksiyonu yardımıyla HTML için listele
        # Format as list for HTML using _format_results function
        formatted_results = self._format_results(final_sorted_table, num)
        
        return formatted_results, warnings

    # =========================================================================
    # Film Detayları / Movie Details
    # =========================================================================

    def get_movie_details(self, movie_id):
        """
        Film detaylarını (Özet, Oyuncular, Tür vb.) yerel veritabanından veya eksikse internetten (API) alır.
        Gets movie details (Overview, Cast, Genre etc.) from local DB or from internet (API) if missing.
        """
        # 1. Girilen ID değerini doğrula / 1. Validate entered ID value
        try:
            valid_movie_id = int(movie_id)
        except (ValueError, TypeError):
            logging.error(f"Geçersiz film ID / Invalid movie ID format: {movie_id}")
            return None

        movie_data_dict = None
        data_source = "unknown"

        # 2. Önce kendi yerel tablomuza (Local Data) bakalım
        # 2. Let's look at our local table first (Local Data)
        has_id_column = ('id' in self.movies_df.columns)
        
        if has_id_column and pd.api.types.is_integer_dtype(self.movies_df['id']):
            # Veritabanında (Pandas DataFrame) bu ID'ye sahip satırları bul
            # Find rows with this ID in database
            local_matches = self.movies_df[self.movies_df['id'] == valid_movie_id]
            
            # Eğer eşleşen bir sonuç (film) varsa / If there is a matching result (movie)
            if not local_matches.empty:
                # İlk eşleşen filmin verilerini bir Sözlüğe (Dictionary) çevir
                # Convert data of first matching movie to a Dictionary
                movie_data_dict = local_matches.iloc[0].to_dict()
                data_source = "local_enhanced"

                # Ekstra Güzellik: Yerel verideki İngilizce özet yerine API'den güncel Türkçe özet getirmeye çalış
                # Extra Polish: Try to get fresh Turkish overview from API instead of local English overview
                if TMDB_API_KEY:
                    try:
                        api_request_url = f"{TMDB_API_BASE_URL}/movie/{valid_movie_id}?api_key={TMDB_API_KEY}&language=tr-TR"
                        api_response = requests.get(api_request_url, timeout=API_TIMEOUT)
                        api_response.raise_for_status()
                        
                        online_movie_data = api_response.json()
                        
                        # Türkçe açıklamaları (özet ve slogan) güncelle / Update Turkish descriptions
                        if online_movie_data.get('overview'):
                            movie_data_dict['overview'] = online_movie_data['overview']
                        if online_movie_data.get('tagline'):
                            movie_data_dict['tagline'] = online_movie_data['tagline']
                        if online_movie_data.get('runtime'):
                            movie_data_dict['runtime'] = online_movie_data['runtime']
                    except Exception as error:
                        logging.warning(f"API Türkçe detay çekilirken hata / API Turkish detail error: {error}")

        # 3. Kendi veritabanımızda film yoksa, doğrudan API'den (İnternetten) her şeyi çek
        # 3. If movie not in our DB, fetch everything directly from API (Internet)
        is_movie_missing = (movie_data_dict is None)
        
        if is_movie_missing and TMDB_API_KEY:
            try:
                # API adresini oluştur. Ekstra olarak 'credits' (oyuncular) ve 'videos' (fragmanlar) da istiyoruz.
                # Create API address. Additionally requesting 'credits' (cast) and 'videos' (trailers).
                api_full_url = (
                    f"{TMDB_API_BASE_URL}/movie/{valid_movie_id}"
                    f"?api_key={TMDB_API_KEY}&language=tr-TR"
                    f"&append_to_response=credits,videos"
                )
                
                full_response = requests.get(api_full_url, timeout=API_TIMEOUT)
                full_response.raise_for_status()
                online_full_data = full_response.json()

                # API'den gelen karmaşık Tür (Genre) verilerini kendi sistemimize uyarla
                # Adapt complex Genre data from API to our system
                formatted_genre_details = []
                english_genre_names_list = []

                for genre_item in online_full_data.get('genres', []):
                    g_id = genre_item.get('id')
                    g_name_tr = genre_item.get('name') # Türkçe isim API'den gelir
                    
                    if g_id and g_name_tr:
                        # Haritamızdan İngilizce eşdeğerini bul / Find English equivalent from our map
                        en_name = self.genre_id_to_english_name_map.get(g_id, g_name_tr)
                        
                        # Yasaklı bir tür değilse listelere ekle / Add if not an excluded genre
                        if en_name not in EXCLUDED_GENRES_FOR_POPULATE:
                            formatted_genre_details.append({
                                'id': g_id,
                                'display_name': g_name_tr,
                                'link_name': en_name,
                            })
                            english_genre_names_list.append(en_name)

                # Çıkış tarihinden yılı (İlk 4 harf) kopar / Extract year from release date
                raw_release_date = online_full_data.get('release_date', '')
                extracted_year = int(raw_release_date[:4]) if raw_release_date else 0

                # Oyuncu listesinin sadece en üstteki 15 oyuncusunu al
                # Take only top 15 actors of cast list
                credits_dict = online_full_data.get('credits', {})
                cast_raw_list = credits_dict.get('cast', [])[:15]
                
                actor_names_only = []
                for actor in cast_raw_list:
                    actor_names_only.append(actor['name'])
                    
                final_cast_string = ', '.join(actor_names_only) # İsimleri virgülle birleştir

                # Yönetmeni (Director) ekip (crew) arasından bul
                # Find Director among crew
                movie_director = ''
                crew_raw_list = credits_dict.get('crew', [])
                for crew_member in crew_raw_list:
                    if crew_member.get('job') == 'Director':
                        movie_director = crew_member.get('name', '')
                        break # Yönetmeni bulduk, döngüden çık / Found director, break loop

                # API'den aldığımız yeni bilgilerle kendi sistemimize uygun bir sözlük (Film) oluştur
                # Create a database-compatible dictionary (Movie) with new info from API
                movie_data_dict = {
                    'id': online_full_data.get('id'),
                    'title': online_full_data.get('title'),
                    'overview': online_full_data.get('overview'),
                    'tagline': online_full_data.get('tagline'),
                    'genres': ' '.join(sorted(set(english_genre_names_list))),
                    'genre_details_for_display': formatted_genre_details,
                    'year': extracted_year,
                    'vote_average': online_full_data.get('vote_average'),
                    'vote_count': online_full_data.get('vote_count'),
                    'popularity': online_full_data.get('popularity'),
                    'cast': final_cast_string,
                    'director': movie_director,
                    'keywords': '', # API'den kelimeleri almak zorund değiliz, boş bırakıyoruz
                    'runtime': online_full_data.get('runtime'),
                }
                data_source = "tmdb_api_only" # Verinin internetten geldiğini işaretle

            except requests.exceptions.HTTPError as http_error:
                # 404 hatası film bulunamadı demektir, çökme yapmadan dön
                # 404 means movie not found, return without crashing
                if http_error.response.status_code == 404:
                    return None
                logging.error(f"API bağlanma hatası / API HTTP error: {http_error}")
                return None
            except Exception as e:
                logging.error(f"Beklenmeyen API hatası / Unexpected API error: {e}")
                return None

        # 4. Hala film bulamadıysak boş dön
        # 4. If still no movie found, return empty
        if movie_data_dict is None:
            return None

        # 5. Türlerin Ekranda Düzgün Görünmesi İçin Son Kontroller (Eğer lokalden geldiyse)
        # 5. Final Checks for Proper Display of Genres (If it came from local)
        if data_source == "local_enhanced" and 'genre_details_for_display' not in movie_data_dict:
            local_genre_display = []
            raw_genres_string = movie_data_dict.get('genres', '')
            
            if isinstance(raw_genres_string, str):
                for single_genre in raw_genres_string.split():
                    clean_genre = single_genre.strip()
                    if clean_genre and clean_genre.lower() not in EXCLUDED_GENRES_FOR_POPULATE:
                        # Haritadan ID'sini bulup sözlük oluştur
                        found_genre_id = self.english_genre_name_to_id_map.get(clean_genre.lower())
                        local_genre_display.append({
                            'id': found_genre_id,
                            'display_name': clean_genre,
                            'link_name': clean_genre, # Tıklanabilir link için / For clickable link
                        })
            movie_data_dict['genre_details_for_display'] = local_genre_display

        # 6. Afiş (Poster) ve Fragman (Trailer) Ekle
        # 6. Add Poster and Trailer
        # Kalite önceliği: Büyük boy > Küçük boy
        # Quality priority: Large size > Small size
        movie_data_dict['poster_url_large'] = (
            self.get_movie_poster_url(valid_movie_id, size='w500') or
            self.get_movie_poster_url(valid_movie_id, size='w342')
        )
        movie_data_dict['trailer_key'] = self.get_movie_videos(valid_movie_id)

        # 7. Arayüze Hata Vermemesi İçin Sayıları ve Metinleri Temizle
        # 7. Clean Numbers and Texts to Prevent UI Errors
        
        # Yıl Temizliği / Year Clean
        raw_year = movie_data_dict.get('year', 0)
        movie_data_dict['year'] = int(raw_year) if pd.notna(raw_year) else 0

        # Oy Puanı Temizliği / Vote Score Clean
        raw_vote_avg = movie_data_dict.get('vote_average', 0.0)
        movie_data_dict['vote_average'] = round(float(raw_vote_avg), 1) if pd.notna(raw_vote_avg) else 0.0

        # Oy Sayısı Temizliği / Vote Count Clean
        raw_vote_cnt = movie_data_dict.get('vote_count', 0)
        movie_data_dict['vote_count'] = int(raw_vote_cnt) if pd.notna(raw_vote_cnt) else 0

        # Popülerlik Temizliği / Popularity Clean
        raw_pop = movie_data_dict.get('popularity', 0.0)
        movie_data_dict['popularity'] = float(raw_pop) if pd.notna(raw_pop) else 0.0

        # Metin (Yazı) Alanlarını String'e Çevir (None dönmesini engeller)
        # Convert Text Fields to String (Prevents returning None)
        text_fields_list = ['title', 'overview', 'genres', 'keywords', 'cast', 'director', 'tagline']
        for text_key in text_fields_list:
            movie_data_dict[text_key] = str(movie_data_dict.get(text_key, ''))

        # Düşük İhtimal: Tür detay listesi oluşmadıysa, olan metinle zorla bir liste yap
        # Edge Case: If genre details list wasn't created, force a list with existing text
        if not movie_data_dict.get('genre_details_for_display'):
            fallback_genres_list = []
            for fallback_genre in movie_data_dict.get('genres', '').split():
                clean_fb_genre = fallback_genre.strip()
                if clean_fb_genre:
                    fallback_genres_list.append({
                        'id': self.english_genre_name_to_id_map.get(clean_fb_genre.lower()),
                        'display_name': clean_fb_genre,
                        'link_name': clean_fb_genre,
                    })
            movie_data_dict['genre_details_for_display'] = fallback_genres_list

        # 8. Süre (Runtime) hesabını arayüzde okunabilir yapmak (Örn: 135 -> "2 sa 15 dk")
        # 8. Make Runtime human readable for UI (E.g: 135 -> "2 h 15 min")
        toplam_dakika_saf = movie_data_dict.get('runtime')
        movie_data_dict['runtime_formatted'] = None
        
        is_runtime_valid = (toplam_dakika_saf is not None and str(toplam_dakika_saf).strip() and str(toplam_dakika_saf).lower() != 'nan')
        
        if is_runtime_valid:
            try:
                toplam_dakika = int(float(toplam_dakika_saf))
                if toplam_dakika > 0:
                    saat_kismi = toplam_dakika // 60
                    dakika_kismi = toplam_dakika % 60
                    
                    if saat_kismi > 0 and dakika_kismi > 0:
                        movie_data_dict['runtime_formatted'] = f"{saat_kismi} sa {dakika_kismi} dk"
                    elif saat_kismi > 0:
                        movie_data_dict['runtime_formatted'] = f"{saat_kismi} sa"
                    else:
                        movie_data_dict['runtime_formatted'] = f"{dakika_kismi} dk"
            except ValueError:
                pass # Hatalı değer gelirse boş bırak (None) / Leave empty (None) if faulty value

        # 9. Filmin sayfasında en alt kısımda gösterilecek 'Benzer Filmler' listesini oluştur
        # 9. Build 'Similar Movies' list to be shown at bottom of movie page
        try:
            if data_source == "local_enhanced":
                # Kendi AI motorumuzu kullanarak benzerleri tavsiye et
                # Use our own AI engine to recommend similars
                similar_movies_list, _ = self.recommend_by_similarity(
                    movie_title_query=movie_data_dict['title'], 
                    num=8, 
                    exclude_movie_id=valid_movie_id
                )
                movie_data_dict['similar_movies'] = similar_movies_list
                
            elif data_source == "tmdb_api_only" and TMDB_API_KEY:
                # Veritabanımızda olmayan bir filmse, API'den internet üstünden benzer istet
                # If movie not in our DB, request similars online from API
                movie_data_dict['similar_movies'] = self._get_similar_from_api(valid_movie_id)
                
            else:
                movie_data_dict['similar_movies'] = []
                
        except Exception as error:
            logging.error(f"Benzer filmleri eklerken hata / Error adding similar movies: {error}")
            movie_data_dict['similar_movies'] = []

        return movie_data_dict


    def _get_similar_from_api(self, movie_id):
        """
        Yerel veritabanında olmayan filmler için, internet üzerinden (TMDb API) benzer film listesi çeker.
        Gets similar movies list online (TMDb API) for movies not in local database.
        """
        try:
            # Benzer filmler adresi / URL for similar movies
            api_url = f"{TMDB_API_BASE_URL}/movie/{movie_id}/similar?api_key={TMDB_API_KEY}&language=tr-TR&page=1"
            response = requests.get(api_url, timeout=API_TIMEOUT)
            response.raise_for_status()
            
            # API'den dönen listeden filmleri al
            # Get movies from API returned list
            api_results = response.json().get('results', [])

            formatted_similar_movies = []
            
            # Sadece 8 tane gösterilecek, her biri için dön
            # Iterate for each, max 8 will be shown
            for movie_obj in api_results[:8]:
                # Gelen film paketinin ismi veya ID'si yoksa onu hiç alma
                # Ignore if incoming movie package has no name or ID
                if not movie_obj.get('id') or not movie_obj.get('title'):
                    continue

                # Posteri bul veya oluştur
                # Find or create poster
                api_poster_path = movie_obj.get('poster_path')
                
                if api_poster_path:
                    # Poster varsa tam link yap / Make full link if exist
                    poster_full_link = f"{TMDB_POSTER_BASE_URL}w342{api_poster_path}"
                else:
                    # Poster yoksa bizim cache fonksiyonumuzla bulmayı dene
                    # If no poster, try to find with our cache function
                    poster_full_link = self.get_movie_poster_url(movie_obj.get('id'), size='w342')

                # Sadece yılı çek / Extract only year
                raw_release_date = movie_obj.get('release_date', '')
                movie_year = int(raw_release_date[:4]) if raw_release_date else 0

                # HTML şablonuna uyan minik bir sözlük oluştur
                # Create a small dictionary compatible with HTML template
                formatted_similar_movies.append({
                    'id': movie_obj.get('id'),
                    'title': movie_obj.get('title'),
                    'poster_url': poster_full_link,
                    'year': movie_year,
                })

            return formatted_similar_movies
            
        except Exception as error:
            logging.warning(f"Benzer film API hatası (Sessizce geçildi) / Similar movies API error (Silently ignored): {error}")
            return []

    # =========================================================================
    # Yardımcı: Film bilgisi çek / Helper: Fetch movie info
    # =========================================================================

    def get_movie_info_for_list(self, movie_id):
        """
        Kişisel sayfa (Beğeniler ve İzleme listesi) ekranları için
        kısaltılmış mini film bilgisini çeker.
        
        Fetches shortened mini movie info for 
        personal page (Likes and Watchlist) screens.
        """
        mini_movie_data = None

        # 1. Önce Hızlı Liste (Local Data) Araması
        # 1. Fast List (Local Data) Search First
        has_id_column = not self.movies_df.empty and 'id' in self.movies_df.columns
        
        if has_id_column:
            # Aradığımız ID tabloda var mı? / Is the ID we are looking for in table?
            local_match_rows = self.movies_df[self.movies_df['id'] == int(movie_id)]
            
            if not local_match_rows.empty:
                # Satırdan bilgileri koparıp sözlük yap
                # Tear info from row and make dictionary
                row_data = local_match_rows.iloc[0].to_dict()
                movie_title = row_data.get('title')
                
                if movie_title:
                    # Yıl bilgisini hatalardan (NaN) arındır
                    # Clean year info from errors (NaN)
                    raw_year = row_data.get('year', 0)
                    clean_year = int(raw_year) if pd.notna(raw_year) else 0

                    # Puan ortalamasını yuvarla (Örn: 5.6)
                    # Round vote average (E.g: 5.6)
                    raw_vote_avg = row_data.get('vote_average', 0.0)
                    clean_vote_avg = round(float(raw_vote_avg), 1) if pd.notna(raw_vote_avg) else 0.0

                    # Mini sözlüğü doldur
                    # Fill mini dictionary
                    mini_movie_data = {
                        'id': int(movie_id),
                        'title': movie_title,
                        'year': clean_year,
                        'vote_average': clean_vote_avg,
                        'poster_url_large': self.get_movie_poster_url(int(movie_id), size='w342'),
                    }

        # 2. Yerelde hiç bilgi yoksa, mecburen API'ye başvuru (Uzun Süreli)
        # 2. If completely no info local, forced to request from API (Long Lasting)
        if mini_movie_data is None and TMDB_API_KEY:
            try:
                # API adresini mini bilgi getirecek şekilde oluştur
                # Create API address to bring mini info
                api_request_address = f"{TMDB_API_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&language=tr-TR"
                api_response = requests.get(api_request_address, timeout=API_TIMEOUT)
                api_response.raise_for_status()
                
                online_info = api_response.json()
                movie_title = online_info.get('title')
                
                if movie_title:
                    # Posteri direkt dönen veriden kurmaya çalış / Try to build poster directly from returned data
                    poster_half_path = online_info.get('poster_path')
                    
                    if poster_half_path:
                        full_poster_url = f"{TMDB_POSTER_BASE_URL}w342{poster_half_path}"
                    else:
                        full_poster_url = self.get_movie_poster_url(int(movie_id), size='w342') # Fonksiyondan son çare / Last resort from function

                    # Yıl çıkar / Extract year
                    raw_release_date = online_info.get('release_date', '')
                    clean_year = int(raw_release_date[:4]) if raw_release_date else 0
                    
                    # Puan yuvarla / Round score
                    raw_vote_avg_api = online_info.get('vote_average', 0.0)
                    clean_vote_avg_api = round(float(raw_vote_avg_api), 1)

                    # Mini sözlüğü API'den kur
                    # Build mini dictionary from API
                    mini_movie_data = {
                        'id': int(online_info.get('id')),
                        'title': movie_title,
                        'year': clean_year,
                        'vote_average': clean_vote_avg_api,
                        'poster_url_large': full_poster_url,
                    }
            except Exception as error:
                logging.warning(f"Kişisel liste için API'den bilgi çekerken hata / API error for personal list (ID: {movie_id}): {error}")

        return mini_movie_data
