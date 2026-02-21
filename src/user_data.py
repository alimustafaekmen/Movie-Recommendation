# =============================================================================
# Kullanıcı Veri Yönetimi / User Data Management
# =============================================================================

import os
import json
import logging
from werkzeug.security import generate_password_hash, check_password_hash

# --- Dosya Yolları / File Paths ---

# Proje kök dizini / Project root directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Veri seti klasörü / Dataset folder
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')

# Kullanıcı bilgileri dosyası / User data file
USERS_JSON = os.path.join(DATASET_FOLDER, 'users.json')

# Kullanıcı etkileşimleri dosyası / User interactions file
USER_INTERACTIONS_JSON = os.path.join(DATASET_FOLDER, 'user_interactions.json')


# =============================================================================
# Kullanıcı İşlemleri / User Operations
# =============================================================================

def load_users():
    """
    Kullanıcı verilerini dosyadan yükler.
    Loads user data from file.
    """
    try:
        # Dosya yoksa oluştur / Create file if not exists
        if not os.path.exists(USERS_JSON):
            with open(USERS_JSON, 'w', encoding='utf-8') as file:
                json.dump({}, file)
            return {}

        # Dosyayı oku / Read the file
        with open(USERS_JSON, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # İçerik boşsa boş sözlük döndür / Return empty dictionary if content is empty
            if not content.strip():
                return {}
                
            # JSON formatını Python sözlüğüne çevir / Convert JSON to Python dictionary
            parsed_data = json.loads(content)
            return parsed_data
            
    except IOError as error:
        logging.error(f"Kullanıcı dosyası okuma hatası / User file read error: {error}")
        return {}
    except json.JSONDecodeError as error:
        logging.error(f"Kullanıcı JSON format hatası / User JSON format error: {error}")
        return {}


def save_users(users_data):
    """
    Kullanıcı verilerini dosyaya kaydeder.
    Saves user data to file.
    """
    try:
        with open(USERS_JSON, 'w', encoding='utf-8') as file:
            json.dump(users_data, file, ensure_ascii=False, indent=4)
    except IOError as error:
        logging.error(f"Kullanıcı kaydetme hatası / User save error: {error}")


def create_user(username, password):
    """
    Yeni kullanıcı oluşturur.
    Creates a new user.

    Returns:
        bool: Başarılı ise True (True if successful), aksi halde False (otherwise False)
    """
    users = load_users()

    # Kullanıcı zaten varsa işlemi iptal et / Cancel if user already exists
    if username in users:
        return False

    # Şifreyi güvenli hale getir (Hash) / Secure the password (Hash)
    hashed_password = generate_password_hash(password)

    # Yeni kullanıcıyı ekle / Add the new user
    users[username] = {
        'password_hash': hashed_password
    }
    
    # Kullanıcıları dosyaya kaydet / Save users to file
    save_users(users)

    # Yeni kullanıcı için etkileşim (beğeni, vb.) listelerini oluştur
    # Create interaction lists (likes, etc.) for the new user
    interactions = load_all_user_interactions()
    
    if username not in interactions:
        interactions[username] = {
            "liked_movies": [], 
            "watchlist_movies": []
        }
        save_all_user_interactions(interactions)

    return True


def get_user(username):
    """
    Kullanıcıyı döndürür. Yoksa None döner.
    Returns user data. Returns None if not found.
    """
    users = load_users()
    
    # Kullanıcı varsa döndür, yoksa None
    # Return user if exists, else None
    if username in users:
        return users[username]
    else:
        return None


def check_user_password(username, password):
    """
    Şifre doğrulaması yapar.
    Checks password validity.
    
    Returns:
        bool: Şifre doğruysa True, yanlışsa False
    """
    user = get_user(username)
    
    # Kullanıcı bulunamadıysa False dön / Return False if user not found
    if not user:
        return False
        
    # Kaydedilmiş şifre özetini al / Get saved password hash
    saved_password_hash = user.get('password_hash', '')
    
    # Şifreleri karşılaştır / Compare passwords
    is_password_correct = check_password_hash(saved_password_hash, password)
    
    if is_password_correct:
        return True
    else:
        return False


# =============================================================================
# Etkileşim İşlemleri / Interaction Operations
# =============================================================================

def load_all_user_interactions():
    """
    Tüm kullanıcıların etkileşimlerini (beğeni vb.) yükler.
    Loads all user interactions (likes etc).
    """
    try:
        # Dosya yoksa boş olarak oluştur / Create empty if file doesn't exist
        if not os.path.exists(USER_INTERACTIONS_JSON):
            with open(USER_INTERACTIONS_JSON, 'w', encoding='utf-8') as file:
                json.dump({}, file, ensure_ascii=False, indent=4)
            return {}

        # Dosyayı oku / Read the file
        with open(USER_INTERACTIONS_JSON, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # İçerik boşsa boş sözlük dön / Return empty dictionary if content is empty
            if not content.strip():
                return {}
                
            parsed_data = json.loads(content)
            return parsed_data
            
    except IOError as error:
        logging.error(f"Etkileşim dosyası okuma hatası / Interaction file read error: {error}")
        return {}
    except json.JSONDecodeError as error:
        logging.error(f"Etkileşim JSON format hatası / Interaction JSON format error: {error}")
        return {}


def save_all_user_interactions(all_interactions_data):
    """
    Tüm kullanıcıların etkileşimlerini kaydeder.
    Saves all user interactions.
    """
    try:
        with open(USER_INTERACTIONS_JSON, 'w', encoding='utf-8') as file:
            json.dump(all_interactions_data, file, ensure_ascii=False, indent=4)
    except IOError as error:
        logging.error(f"Etkileşim kaydetme hatası / Interaction save error: {error}")


def load_user_interactions(username):
    """
    Belirli bir kullanıcının etkileşimlerini (beğeniler, listeler) yükler.
    Loads interactions (likes, lists) for a specific user.
    """
    all_interactions = load_all_user_interactions()

    # Eğer kullanıcının kaydı yoksa, varsayılan listeleri oluştur
    # If the user has no record, create default lists
    if username not in all_interactions:
        default_interactions = {
            "liked_movies": [], 
            "watchlist_movies": []
        }
        all_interactions[username] = default_interactions
        save_all_user_interactions(all_interactions)

    # Kullanıcının etkileşimlerini döndür / Return user's interactions
    user_interactions = all_interactions.get(username)
    
    # Güvenlik önlemi: Eğer hala None ise varsayılan dön
    # Safety check: If still None, return default
    if not user_interactions:
        return {"liked_movies": [], "watchlist_movies": []}
        
    return user_interactions


def save_user_interactions(username, user_interactions_data):
    """
    Belirli bir kullanıcının etkileşimlerini kaydeder.
    Saves interactions for a specific user.
    """
    all_interactions = load_all_user_interactions()
    
    # Kullanıcının verisini güncelle / Update user's data
    all_interactions[username] = user_interactions_data
    
    # Tüm listeyi kaydet / Save the whole list
    save_all_user_interactions(all_interactions)