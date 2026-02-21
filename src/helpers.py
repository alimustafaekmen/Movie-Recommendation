# =============================================================================
# Yardımcı Fonksiyonlar / Helper Functions
# =============================================================================

import json
import logging
import traceback
import pandas as pd
import requests
from urllib.parse import quote_plus, unquote_plus

from src.config import (
    TMDB_API_KEY, TMDB_API_BASE_URL, TMDB_POSTER_BASE_URL, API_TIMEOUT
)


def parse_json_helper(json_str, key=None, job=None, limit=None):
    """
    JSON metnini (string) listeye veya sözlüğe çevirir ve istenilen bilgiyi alır.
    Parses a JSON string and extracts requested information.

    Args:
        json_str: İşlenecek JSON verisi / JSON data to process
        key: Alınmak istenen anahtar kelime (örn: 'name') / Target key (e.g. 'name')
        job: Aranan iş pozisyonu (örn: 'Director') / Job position (e.g. 'Director')
        limit: Alınacak maksimum eleman sayısı / Max number of items to return

    Returns:
        Metin (str) veya liste (list) / String or list
    """
    
    # 1. Veri tipini kontrol et ve veriyi kullanıma hazır hale getir
    # 1. Check data type and make data ready to use
    
    # Veri zaten liste veya sözlük ise doğrudan 'data' değişkenine ata
    # If data is already a list or dict, assign it directly to 'data'
    is_list = isinstance(json_str, list)
    is_dict = isinstance(json_str, dict)
    
    if is_list or is_dict:
        data = json_str
    else:
        # Veri metin (string) ise çalıştırılacak kısım
        # Handled if data is text (string)
        try:
            # Gelen veri boş ise işlemi sonlandır ve boş değer dön
            # If data is empty, return empty value
            is_empty_string = not json_str
            is_empty_array_string = json_str == '[]'
            is_nan_value = pd.isna(json_str)

            if is_empty_string or is_empty_array_string or is_nan_value:
                # 'key' veya 'job' aranıyorsa boş metin dön, yoksa boş liste dön
                # Return empty string if 'key' or 'job' is searched, else return empty list
                if key != None or job != None:
                    return ""
                else:
                    return []

            # JSON metnini Python verisine dönüştür
            # Convert JSON text to Python data
            data_string = str(json_str)
            data = json.loads(data_string)
            
        except json.JSONDecodeError:
            # JSON çevirme hatası olursa
            if key != None or job != None:
                return ""
            else:
                return []
        except TypeError:
            # Tip hatası olursa
            if key != None or job != None:
                return ""
            else:
                return []

        # Eğer dönüştürülen veri liste değilse, tek elemanlı bir liste yap
        # If converted data is not a list, make it a single-item list
        if not isinstance(data, list):
            data = [data]

    # 2. Hazırlanan veriden istenen bilgiyi çıkar
    # 2. Extract specific information from the prepared data
    try:
        # Durum A: Bir kişiyi 'job' (iş) değerine göre arıyorsak
        # Case A: If searching for a person by 'job' value
        if job != None:
            # Listedeki her bir elemanı teker teker dolaş / Loop through each item
            for item in data:
                # Öğenin 'job' değeri aradığımız değere eşitse ismini döndür
                # Return name if item's job matches
                current_job = item.get('job')
                if current_job == job:
                    found_name = item.get('name', '')
                    return found_name
            # Bulunamazsa boş metin dön / Return empty string if not found
            return ""

        # Durum B: Bütün elemanlardan belirli bir 'key' (anahtar) değerini çekip birleştiriyorsak
        # Case B: If extracting and joining a specific 'key' from all items
        elif key != None:
            names_list = []
            
            # Öğeleri dolaş ve değerleri listeye ekle / Loop through and add values to list
            for item in data:
                current_value = item.get(key)
                if current_value != None:
                    value_as_text = str(current_value).strip()
                    names_list.append(value_as_text)
            
            # Listeyi boşluk ile birleştirir / Join list items with space
            result_text = ' '.join(names_list)
            return result_text

        # Durum C: Sadece belli bir sayıya (limit) kadar olan 'name' değerlerini istiyorsak
        # Case C: If requesting 'name' values up to a certain limit
        elif limit != None:
            names_list = []
            
            # İlk 'limit' sayıdaki elemana bak / Check items up to 'limit'
            for item in data[:limit]:
                current_name = item.get('name')
                if current_name != None:
                    name_as_text = str(current_name).strip()
                    names_list.append(name_as_text)
            
            return names_list

        # Durum D: Özel bir istek yoksa orijinal bilgiyi döndür
        # Case D: If no specific request, return the raw data
        else:
            return data

    except Exception as error:
        # Beklenmeyen bir hata olursa konsola yaz ve boş değer dön
        # Print error to console and return empty value if unexpected error occurs
        logging.warning(f"JSON veri işleme sırasında hata / Error during JSON processing: {error}")
        if key != None or job != None:
            return ""
        else:
            return []


def get_person_details_from_tmdb(person_name_url_encoded, person_type="person"):
    """
    TMDb API'sini kullanarak bir oyuncu veya yönetmenin bilgilerini getirir.
    Fetches details of an actor or director from TMDb API.

    Args:
        person_name_url_encoded: İnternet adresine (URL) uygun hale getirilmiş kişi adı
        person_type: "director" (yönetmen) veya "actor" (oyuncu)

    Returns:
        Kişi bilgilerini içeren sözlük (dict) veya başarısız olursa None
        Dictionary containing person details or None if failed
    """
    
    # API Anahtarı girilmemişse uyarı ver ve iptal et
    # Warn and cancel if API key is missing
    if not TMDB_API_KEY:
        logging.warning("TMDb API anahtarı ayarlanmamış / TMDb API key is not set")
        return None

    # Kişinin adını düzelt (örn: "Christopher+Nolan" -> "Christopher Nolan")
    # Decode person name from URL format
    person_name = unquote_plus(person_name_url_encoded)
    logging.info(f"'{person_name}' için bilgiler aranıyor / Searching info for '{person_name}'...")

    try:
        # ==========================================
        # ADIM 1: Kişiyi Arama / STEP 1: Search Person
        # ==========================================
        
        # Arama yapmak için adresi oluştur
        # Build search URL
        safe_name_for_url = quote_plus(person_name)
        search_url = (
            f"{TMDB_API_BASE_URL}/search/person"
            f"?api_key={TMDB_API_KEY}"
            f"&query={safe_name_for_url}"
            f"&language=tr-TR&include_adult=false"
        )
        
        # İnternetten veriyi adresten indir / Download data from URL
        search_response = requests.get(search_url, timeout=API_TIMEOUT)
        search_response.raise_for_status() # Hata varsa (örn. 404) işlemi kes / Raise error if fails
        
        # Gelecek cevabı sözlük haline çevir ve içindeki 'results' listesini al
        # Parse JSON and get 'results' list
        search_data = search_response.json()
        search_results_list = search_data.get('results')

        # Sonuç yoksa çık / Exit if no results
        if not search_results_list:
            logging.warning(f"'{person_name}' TMDb'de bulunamadı / Person not found on TMDb")
            return None

        # En uygun kişiyi seçmek için değişken oluştur
        # Variable to store best match
        best_match_person = None

        # Yönetmen (Director) veya Oyuncu (Acting) durumlarına göre arama sonuçlarını filtrele
        # Filter search results based on person type
        if person_type == "director":
            for search_item in search_results_list:
                department = search_item.get('known_for_department')
                if department == "Directing":
                    best_match_person = search_item
                    break # Bulunca döngüyü durdur / Stop loop when found
        elif person_type == "actor":
            for search_item in search_results_list:
                department = search_item.get('known_for_department')
                if department == "Acting":
                    best_match_person = search_item
                    break # Bulunca döngüyü durdur / Stop loop when found

        # Özel bölüm filtresinde bulamadıysak en üstteki ilk sonucu kabul et
        # If couldn't find specific department, use first result
        if best_match_person == None and len(search_results_list) > 0:
            best_match_person = search_results_list[0]

        # Eğer hala kişi yoksa veya numarası (ID) yoksa işlemi iptal et
        # Cancel if person is still empty or has no ID
        if best_match_person == None or best_match_person.get('id') == None:
            logging.warning(f"'{person_name}' için geçerli ID yok / No valid ID for person")
            return None

        # Doğru kişinin numarasını al
        person_id = best_match_person.get('id')


        # ==========================================
        # ADIM 2: Detayları Alma / STEP 2: Get Details
        # ==========================================

        # Kişinin kendi özel sayfası için adres oluştur (Ayrıca filmlerini de dahil et)
        # Create URL for person's details including their movies
        detail_url = (
            f"{TMDB_API_BASE_URL}/person/{person_id}"
            f"?api_key={TMDB_API_KEY}"
            f"&language=tr-TR&append_to_response=movie_credits"
        )
        
        # Detayları indir / Download details
        detail_response = requests.get(detail_url, timeout=API_TIMEOUT)
        detail_response.raise_for_status()
        
        # Bilgileri sözlüğe dönüştür / Convert to dictionary
        detailed_data = detail_response.json()

        # Resim yolunu güvenli bir şekilde al / Get profile path safely
        profile_path = detailed_data.get('profile_path')
        
        # Tam resim adresini oluştur (Resim yoksa None bırak)
        # Create full image URL (Leave None if doesn't exist)
        if profile_path != None:
            full_profile_image_url = f"{TMDB_POSTER_BASE_URL}h632{profile_path}"
        else:
            full_profile_image_url = None

        # Biyografi kontrolü (Boşsa varsayılan yazı yaz)
        # Biography check
        biography_text = detailed_data.get('biography')
        if not biography_text:
            biography_text = "Biyografi bilgisi bulunmamaktadır."

        # Sayfaya gönderilecek temel kişi bilgilerini bir sözlükte topla
        # Collect basic person details in a dictionary
        person_info_result = {
            'id': detailed_data.get('id'),
            'name': detailed_data.get('name'),
            'biography': biography_text,
            'birthday': detailed_data.get('birthday'),
            'deathday': detailed_data.get('deathday'),
            'place_of_birth': detailed_data.get('place_of_birth'),
            'profile_path': profile_path,
            'known_for_department': detailed_data.get('known_for_department'),
            'profile_image_url': full_profile_image_url,
            'homepage': detailed_data.get('homepage'),
        }


        # ==========================================
        # ADIM 3: Filmleri İşleme / STEP 3: Process Movies
        # ==========================================

        movies_list = []
        # 'movie_credits' bölümü filmleri içerir
        raw_movie_credits = detailed_data.get('movie_credits', {})

        # 'director' (yönetmen) isen arka plan (crew) ekibi listesine bakarız,
        # 'actor' (oyuncu) isen oyuncular (cast) listesine bakarız.
        if person_type == "director":
            credits_array = raw_movie_credits.get('crew', [])
        else:
            credits_array = raw_movie_credits.get('cast', [])

        # Sadece resimli ve doğru görevli filmleri seçeceğimiz liste
        # List to keep valid movies
        valid_movies_list = []
        
        for single_movie in credits_array:
            # Sadece afişi olanları kabul et / Only accept movies with a poster
            poster = single_movie.get('poster_path')
            if not poster:
                continue
                
            # Eğer bu kişi yönetmense, görev (job) içeriği mutlaka 'Director' olmalı.
            # If person is director, job must be 'Director'
            if person_type == "director":
                job_title = single_movie.get('job')
                if job_title != 'Director':
                    continue
                    
            valid_movies_list.append(single_movie)

        # Filmleri önce tarihe(yeni olana), sonra popülerliğe göre sıraya sokacak fonksiyon
        # Function to sort movies by date and popularity
        def sort_key_function(movie_item):
            release_date = movie_item.get('release_date')
            popularity = movie_item.get('popularity', 0)
            
            # Tarih hatalı ise veya yoksa en küçük tarihi ver (listede sona atar)
            # Give min date if date is empty or invalid
            try:
                date_object = pd.to_datetime(release_date) if release_date else pd.Timestamp.min
            except ValueError:
                date_object = pd.Timestamp.min
                
            return (date_object, popularity)

        # Listeyi bu fonksiyona göre büyükten küçüğe (reverse=True) sırala
        # Sort list descending
        valid_movies_list.sort(key=sort_key_function, reverse=True)

        # Ekranda göstereceğimiz son film listesini hazırla
        # Prepare the final movie list for display
        added_movie_ids = set() # Daha önce eklenen filmleri hatırlamak için
        
        for movie_item in valid_movies_list:
            current_movie_id = movie_item.get('id')

            # Bu film daha önce eklendiyse tekrar ekleme / Skip if already added
            if current_movie_id in added_movie_ids:
                continue

            # Özet karakter sayısını 120 ile sınırla
            # Limit overview to 120 characters
            overview_text = movie_item.get('overview', '')
            if overview_text:
                short_overview = overview_text[:120] + "..."
            else:
                short_overview = "Özet bulunmuyor."

            # Film bilgilerini düzenle / Organize movie properties
            poster_endpoint = movie_item.get('poster_path')
            title_text = movie_item.get('title')
            # Eğer başlık yoksa 'name' isimli diğer özelliğe bak
            if not title_text:
                title_text = movie_item.get('name')

            movie_dictionary = {
                'id': current_movie_id,
                'title': title_text,
                'poster_url': f"{TMDB_POSTER_BASE_URL}w342{poster_endpoint}",
                'release_date': movie_item.get('release_date'),
                'overview': short_overview,
                'vote_average': movie_item.get('vote_average', 0),
            }

            # Eğer yönetmense karta "Yönetmen" yazısını ekle
            if person_type == "director":
                movie_dictionary['job'] = 'Yönetmen'
                
            # Eğer oyuncuysa karta oynadığı karakterin adını ekle
            elif person_type == "actor":
                character_name = movie_item.get('character', 'Bilinmiyor')
                movie_dictionary['character'] = character_name

            # Filmi nihai listemize ekle
            movies_list.append(movie_dictionary)
            added_movie_ids.add(current_movie_id)

            # Sadece 24 tane film göstermek için limiti kontrol et
            # Limit the output to 24 movies
            if len(movies_list) >= 24:
                break

        # Filmleri ve diğer tüm bilgileri birleştir
        # Merge all data
        person_info_result['movies'] = movies_list
        
        # Ekranda (HTML) gösterilecek Türkçe meslek ismi
        if person_type == "director":
            person_info_result['person_type_tr'] = "Yönetmen"
        else:
            person_info_result['person_type_tr'] = "Oyuncu"
            
        person_info_result['page_title_name'] = person_name

        logging.info(f"'{person_name}' için toplam {len(movies_list)} adet geçerli film bulundu.")
        return person_info_result

    except requests.exceptions.RequestException as error:
        logging.error(f"İnternet bağlantı veya API hatası / API connection error for '{person_name}': {error}")
        return None
    except Exception as error:
        logging.error(f"Sistemde genel bir hata oluştu / General error for '{person_name}': {error}\n{traceback.format_exc()}")
        return None
