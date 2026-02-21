# =============================================================================
# Film Öneri Sistemi - Ana Uygulama / Movie Recommendation System - Main App
# =============================================================================

import datetime
import logging
import pandas as pd
from flask import Flask, request, render_template, url_for, abort, jsonify, redirect, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user, AnonymousUserMixin
from urllib.parse import quote_plus, unquote_plus

from src import user_data
from src.config import (
    FLASK_SECRET_KEY, MOVIES_CSV, CREDITS_CSV,
    DEFAULT_RECOMMENDATION_COUNT, MAX_RECOMMENDATION_COUNT,
    TMDB_API_KEY
)
from src.helpers import get_person_details_from_tmdb
from src.recommendation import RecommendationEngine


# =============================================================================
# Flask Uygulaması Kurulumu / Flask App Setup
# =============================================================================

# Loglama ayarları: Uygulama çalışırken hataları ve bilgileri konsola yazdırır.
# Logging settings: Prints errors and info messages to console while app is running.
# Renkli Log Formatter Sınıfı / Colored Log Formatter Class
class ColoredFormatter(logging.Formatter):
    """
    Log mesajlarını renklendiren özel formatter.
    Custom formatter to colorize log messages.
    """
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Log formatı: Daha temiz ve okunaklı bir yapı / Cleaner and more readable structure
    # Format: [10:25:34] INFO: Message
    format_str = "[%(asctime)s] %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

# Ana Logger Ayarları / Main Logger Settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Eski handler'ları temizle (varsa) / Clear old handlers (if any)
if logger.hasHandlers():
    logger.handlers.clear()

# Konsol Handler Ekle / Add Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# Varsayılan Flask (Werkzeug) loglarını kapat / Suppress default Flask logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)


# Flask uygulamasını oluştur / Create Flask application
app = Flask(__name__)

# Oturum güvenliği için gizli anahtar (cookie şifreleme).
# Secret key for session security (cookie encryption).
app.secret_key = FLASK_SECRET_KEY

# Flask-Login: Kullanıcı giriş/çıkış yönetimi için eklenti.
# Flask-Login: Extension for managing user login/logout.
login_manager = LoginManager()
login_manager.init_app(app)

# Giriş yapmadan korumalı sayfaya erişildiğinde yönlendirilecek sayfa.
# Page to redirect to when accessing a protected page without login.
login_manager.login_view = 'login_route'

# Giriş yapılmadığında gösterilecek uyarı mesajı.
# Warning message shown when user is not logged in.
login_manager.login_message = "Bu sayfayı görüntülemek için giriş yapın."
login_manager.login_message_category = "info"


# İstek Loglama (Sadeleştirilmiş) / Request Logging (Simplified)
@app.after_request
def log_request(response):
    """Her istekten sonra sade bir log bas / Log after each request"""
    if request.path.startswith('/static'):
        return response  # Statik dosyaları loglama (isteğe bağlı) / Don't log static files
    
    # Renkli çıktı için / For colored output
    cyan = "\x1b[36;20m"
    reset = "\x1b[0m"
    
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    logging.info(f"{cyan}{request.method} {request.path} ({response.status_code}){reset}")
    return response


# =============================================================================
# Kullanıcı Modeli / User Model
# =============================================================================

class User(UserMixin):
    """Kullanıcı sınıfı / User class"""

    def __init__(self, username):
        self.username = username
        self.id = username  # Flask-Login için gerekli / Required for Flask-Login

    @staticmethod
    def get(user_id):
        """Kullanıcıyı veritabanından al / Get user from database"""
        user_info = src.user_data.get_user(user_id)
        if user_info:
            return User(user_id)
        return None


@login_manager.user_loader
def load_user(user_id):
    """Flask-Login kullanıcı yükleyici / Flask-Login user loader"""
    return User.get(user_id)


@app.context_processor
def inject_user():
    """Tüm template'lere kullanıcıyı ekle / Inject user into all templates"""
    try:
        return {'current_user': current_user}
    except NameError:
        return {'current_user': AnonymousUserMixin()}


# =============================================================================
# Öneri Motorunu Başlat / Initialize Recommendation Engine
# =============================================================================

try:
    engine = RecommendationEngine(MOVIES_CSV, CREDITS_CSV)
except Exception as e:
    logging.critical(f"Motor başlatılamadı / Engine failed to start: {e}")
    engine = None


# =============================================================================
# Jinja2 Filtreler / Jinja2 Filters
# =============================================================================

@app.template_filter('format_date')
def format_date_filter(value, format_str='%d %B %Y'):
    """
    Tarihi Türkçe formata çevirir / Formats date to Turkish
    Örn/Example: '2023-10-26' -> '26 Ekim 2023'
    """
    if not value or str(value).lower() == 'nan' or str(value).strip() == '':
        return "Bilinmiyor"
    try:
        date_obj = pd.to_datetime(value).to_pydatetime()

        # İngilizce -> Türkçe ay isimleri / English -> Turkish month names
        months_tr = {
            "January": "Ocak", "February": "Şubat", "March": "Mart",
            "April": "Nisan", "May": "Mayıs", "June": "Haziran",
            "July": "Temmuz", "August": "Ağustos", "September": "Eylül",
            "October": "Ekim", "November": "Kasım", "December": "Aralık"
        }

        formatted = date_obj.strftime(format_str)
        for en_month, tr_month in months_tr.items():
            if en_month in formatted:
                return formatted.replace(en_month, tr_month)
        return formatted
    except Exception:
        return value


# =============================================================================
# Ana Sayfa / Home Page
# =============================================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    """Ana sayfa / Home page
    Bu fonksiyon ana sayfayı gösterir ve kullanıcının film aramalarını/filtrelemelerini yönetir.
    This function renders the home page and handles user movie searches/filtering.
    """

    # 1. Öneri motoru (engine) çalışmıyorsa veya veri yoksa iptal et
    # 1. Cancel if recommendation engine is not running or no data
    is_engine_ready = (engine is not None)
    
    if not is_engine_ready or engine.movies_df.empty:
        return render_template('error.html', message="Film öneri motoru kullanılamıyor. Lütfen daha sonra tekrar deneyin."), 503

    # 2. Şablona (Template) gönderilecek varsayılan boş değişkenleri oluştur
    # 2. Create default empty variables to send to the template
    suggestions = None
    search_term = None
    warnings = []
    recommendation_type = None
    form_data = {}
    selected_genres = []

    # ==========================================
    # DURUM A: Eğer sayfaya GET isteği (URL üzerinden) geldiyse
    # CASE A: If the page received a GET request (via URL)
    # ==========================================
    is_get_request = (request.method == 'GET')
    genre_in_url = request.args.get('genre')

    if is_get_request and genre_in_url:
        # URL'deki 'genre' kelimesini düzelt (boşlukları ayarlamak vs. için)
        # Decode the genre word in URL
        genre_from_url = unquote_plus(genre_in_url)
        
        # Seçili tür listesine ekle / Add to selected genres list
        selected_genres = [genre_from_url]
        form_data['tur'] = selected_genres
        recommendation_type = 'features' # Özellik tabanlı (filtreli) arama yapıyoruz / Feature based search

        # Öneri motorundan türe göre film önerilerini al
        # Get movie recommendations by genre from the engine
        suggestions, warnings = engine.recommend_by_features(
            genres=selected_genres,
            num=DEFAULT_RECOMMENDATION_COUNT
        )
        
        # Ekranda gösterilecek arama terimi başlığını hazırla
        # Prepare the search term title for display
        search_term = f"Tür: {selected_genres[0].capitalize()}"

    # ==========================================
    # DURUM B: Eğer form butonuna basılarak POST isteği geldiyse
    # CASE B: If the page received a POST request (form button clicked)
    # ==========================================
    is_post_request = (request.method == 'POST')

    if is_post_request:
        # Formdaki tüm verileri sözlük (dict) olarak al
        # Get all form data as a dictionary
        form_data = request.form.to_dict()
        
        # Kullanıcının seçtiği türleri (birden fazla olabilir) liste olarak al
        # Get selected genres (can be multiple) as a list
        selected_genres = request.form.getlist('tur')
        form_data['tur'] = selected_genres

        try:
            # Kaç tane film önerisi istendiğini al (Eğer girilmediyse varsayılanı kullan)
            # Get requested number of recommendations (Use default if not provided)
            requested_num = request.form.get('num_recommendations', DEFAULT_RECOMMENDATION_COUNT)
            num_recs = int(requested_num)
            
            # Girilen sayının minimum 1, maksimum MAX_RECOMMENDATION_COUNT arasında olmasını sağla
            # Ensure number is between 1 and MAX_RECOMMENDATION_COUNT
            if num_recs < 1:
                num_recs = 1
            elif num_recs > MAX_RECOMMENDATION_COUNT:
                num_recs = MAX_RECOMMENDATION_COUNT

            # === Seçenek 1: 'Benzer Film Bul' butonuna basılmışsa ===
            # === Option 1: If 'Find Similar Movie' button is clicked ===
            if 'submit_similarity' in request.form:
                recommendation_type = 'similarity'

                # Kullanıcı serbest metin kutusunu mu işaretledi?
                # Did user check the free text input box?
                is_custom_movie_checked = (request.form.get('toggle_film') == 'on')
                
                if is_custom_movie_checked:
                    movie_query = request.form.get('film_ismi_custom')
                else:
                    movie_query = request.form.get('film_ismi')

                # Film ismi varsa başındaki ve sonundaki boşlukları temizle
                # Clean spaces if movie name exists
                if movie_query:
                    search_term = movie_query.strip()
                else:
                    search_term = None

                if search_term:
                    # Motor aracılığıyla benzer filmleri getir
                    # Fetch similar movies via engine
                    suggestions, warnings = engine.recommend_by_similarity(search_term, num=num_recs)
                else:
                    warnings.append("Lütfen bir film ismi girin veya listeden seçin.")

            # === Seçenek 2: 'Filtrelere Göre Öner' butonuna basılmışsa ===
            # === Option 2: If 'Recommend by Filters' button is clicked ===
            elif 'submit_features' in request.form:
                recommendation_type = 'features'

                # Kullanıcı kendi yönetmenini mi yazdı?
                # Did user type their own director?
                is_custom_director_checked = (request.form.get('toggle_yonetmen') == 'on')

                if is_custom_director_checked:
                    director_q = request.form.get('yonetmen_custom')
                else:
                    director_q = request.form.get('yonetmen')
                    
                # Yönetmen ismi varsa boşlukları temizle / Clean spacing
                if director_q:
                    director_q = director_q.strip()
                else:
                    director_q = None

                # Diğer filtre verilerini al / Get other filter data
                min_r = request.form.get('min_puan', '')
                min_y = request.form.get('min_yil', '')
                max_y = request.form.get('max_yil', '')

                # ----------------------------------------
                # Arama Terimi Başlığını Oluşturma (Ekranda görünmesi için)
                # Build Search Term Title (For displaying on screen)
                # ----------------------------------------
                # Başlığı oluşturacak parçaları tutacağımız liste / List to hold title parts
                title_parts = []
                
                # Tür seçilmişse / If genre selected
                has_selected_genres = len(selected_genres) > 0
                if has_selected_genres:
                    genre_names_capitalized = []
                    for genre_item in selected_genres:
                        # İlk harfini büyük yap (Aksiyon vb.) / Capitalize first letter
                        genre_names_capitalized.append(genre_item.capitalize())
                    
                    # Türleri virgülle birleştir ve başlık parçalarına ekle
                    # Join genres with comma and add to title parts
                    combined_genres_string = ', '.join(genre_names_capitalized)
                    title_parts.append(f"Türler: {combined_genres_string}")
                    
                # Yönetmen seçilmişse / If director selected
                if director_q:
                    title_parts.append(f"Yönetmen: {director_q}")
                    
                # Yıl aralığı seçilmişse / If year range selected
                if min_y or max_y:
                    year_features_list = []
                    if min_y: 
                        year_features_list.append(min_y)
                    if max_y: 
                        year_features_list.append(max_y)
                    
                    # Varsa iki yılı da tire (-) ile birleştir / Join both years with dash if exist
                    year_range_string = ' - '.join(year_features_list)
                    title_parts.append(f"Yıl: {year_range_string}")
                    
                # Minimum puan seçilmişse / If minimum rating selected
                if min_r:
                    title_parts.append(f"Min. Puan: {min_r}")
                
                # Oluşan tüm parçaları bir "boru" (|) karakteri ile tek bir cümleye çevir
                # Convert all parts into a single sentence separated by a "pipe" (|) character
                has_title_parts = len(title_parts) > 0
                if has_title_parts:
                    search_term = " | ".join(title_parts)
                else:
                    search_term = "Seçilen Filtreler"

                # ----------------------------------------
                # Önerileri Alma / Fetch Recommendations
                # ----------------------------------------
                # Hazırlanan tüm bu filtreleri motorun özel sayfasına gönder
                # Send all these prepared filters to engine's custom page
                suggestions, warnings = engine.recommend_by_features(
                    genres=selected_genres, 
                    director_query=director_q, 
                    min_rating=min_r, 
                    min_year=min_y, 
                    max_year=max_y, 
                    num=num_recs
                )

        except ValueError:
            # Kullanıcı sayısal değer girmek yerine harf falan girerse hatayı engelle
            # Prevent error if user types letters instead of numbers
            warnings.append("Lütfen sayısal alanlara sadece geçerli sayılar girin.")
        except Exception as e:
            # Bilinmeyen bir hata olursa uygulamayı çökertme
            # Don't crash app if an unknown error occurs
            logging.error(f"Ana sayfa hatası / Home page error: {e}", exc_info=True)
            warnings.append("Öneriler sayfaya yüklenirken bir hata oluştu.")

    # 3. Hazırlanan tüm bilgileri HTML şablonuna (index.html) gönder
    # 3. Send all prepared data to the HTML template (index.html)
    context = {
        'suggestions': suggestions,
        'search_term': search_term,
        'warnings': warnings,
        'recommendation_type': recommendation_type,
        'popular_movies': engine.popular_movies_list,
        'popular_directors': engine.popular_directors_list,
        'all_genres': engine.all_genres,
        'all_movie_titles': engine.all_movie_titles,
        'all_director_names': engine.all_director_names,
        'current_year': datetime.datetime.now().year,
        'form_data': form_data,
        'selected_genres': selected_genres,
        'max_recommendations': MAX_RECOMMENDATION_COUNT,
        'default_recommendations': DEFAULT_RECOMMENDATION_COUNT,
        'quote_plus': quote_plus,
    }
    return render_template('index.html', **context)


# =============================================================================
# Film Detay Sayfası / Movie Detail Page
# =============================================================================

@app.route('/movie/<int:movie_id>')
def movie_details_page(movie_id):
    """
    Film detay sayfası / Movie detail page
    Bir filmin tüm detaylarını (oyuncular, özet, poster vb.) gösterir.
    Shows all details of a movie (cast, overview, poster etc.).
    """

    # 1. Motorun aktif olup olmadığını kontrol et / Check if engine is active
    is_engine_ready = (engine is not None)
    if not is_engine_ready:
        flash("Öneri motoru kullanılamıyor.", "danger")
        return render_template("error.html", error_message="Motor yüklenemedi. / Engine could not be loaded."), 503

    # 2. Film verilerini al / Get movie data
    try:
        movie_data = engine.get_movie_details(movie_id)
        
        # Eğer film veritabanında veya API'da yoksa 404 (Bulunamadı) hatası ver
        # If movie is not in database or API, return 404 (Not Found) error
        if movie_data is None:
            abort(404)
            
    except Exception as error:
        # Teknik bir hata olursa çökmesini engelle / Prevent crash on technical error
        logging.error(f"Film detay hatası / Movie detail error ({movie_id}): {error}")
        return render_template("error.html", error_message=f"Film detayı alınırken hata: {error}"), 500

    # 3. İsteğe Bağlı: Kullanıcı etkileşimlerini (Beğeni durumu vb.) kontrol et
    # 3. Optional: Check user interactions (Like status etc.)
    is_movie_liked = False
    is_movie_watchlisted = False
    
    # Sadece giriş yapmış yetkili kullanıcıların verilerini kontrol ediyoruz
    # We only check data for logged-in authorized users
    is_user_logged_in = current_user.is_authenticated
    if is_user_logged_in:
        # Veritabanından (veya JSON dosyasından) kullanıcının özel listelerini çek
        # Fetch user's private lists from database (or JSON file)
        user_interactions_dict = src.user_data.load_user_interactions(current_user.username)
        
        liked_movies_list = user_interactions_dict.get('liked_movies', [])
        watchlist_movies_list = user_interactions_dict.get('watchlist_movies', [])
        
        # Bu sayfada gösterilen filmin (movie_id) bu listelerde adı (ID'si) var mı yok mu diye bak
        # Check if the movie shown on this page (movie_id) has its name (ID) in these lists
        is_movie_liked = (movie_id in liked_movies_list)
        is_movie_watchlisted = (movie_id in watchlist_movies_list)

    # 4. Hazırlanan tüm verileri arayüze (HTML) gönder / Send all prepared data to UI (HTML)
    return render_template(
        'movie_detail.html',
        movie=movie_data,
        is_liked=is_movie_liked,
        is_watchlisted=is_movie_watchlisted,
        datetime_module=datetime,
        quote_plus=quote_plus,
    )


# =============================================================================
# Tür ve Kişi Sayfaları / Genre and Person Pages
# =============================================================================

@app.route('/genre/<path:genre_name>')
def movies_by_genre(genre_name):
    """Tür sayfası - ana sayfaya yönlendir / Genre page - redirect to home"""
    return redirect(url_for('home', genre=quote_plus(genre_name)))


@app.route('/director/<path:director_name_url>')
def director_details_route(director_name_url):
    """
    Yönetmen detay sayfası / Director detail page
    Seçilen yönetmenin bilgilerini ve filmlerini gösterir.
    Shows details and movies of the selected director.
    """
    is_engine_ready = (engine is not None)
    if not is_engine_ready:
        abort(503) # 503: Servis Kullanılamıyor / Service Unavailable
        
    # Seçili kişinin bilgilerini TMDb API üzerinden getir / Fetch person details from TMDb API
    person_data = get_person_details_from_tmdb(director_name_url, person_type="director")
    
    # Kişi bulunamazsa ana sayfaya dön / Return home if person not found
    is_person_not_found = (person_data is None)
    if is_person_not_found:
        clean_name = unquote_plus(director_name_url)
        flash(f"'{clean_name}' adlı yönetmen bulunamadı. / Director not found.", "warning")
        return redirect(url_for('home'))
        
    # Verileri HTML sayfasına gönder / Send data to HTML page
    return render_template(
        'person_detail.html', 
        person=person_data,
        datetime_module=datetime, 
        quote_plus=quote_plus
    )


@app.route('/actor/<path:actor_name_url>')
def actor_details_route(actor_name_url):
    """
    Oyuncu detay sayfası / Actor detail page
    Seçilen oyuncunun bilgilerini ve filmlerini gösterir.
    Shows details and movies of the selected actor.
    """
    is_engine_ready = (engine is not None)
    if not is_engine_ready:
        abort(503) # 503: Servis Kullanılamıyor / Service Unavailable
        
    # Seçili kişinin bilgilerini TMDb API üzerinden getir / Fetch person details from TMDb API
    person_data = get_person_details_from_tmdb(actor_name_url, person_type="actor")
    
    # Kişi bulunamazsa ana sayfaya dön / Return home if person not found
    is_person_not_found = (person_data is None)
    if is_person_not_found:
        clean_name = unquote_plus(actor_name_url)
        flash(f"'{clean_name}' adlı oyuncu bulunamadı. / Actor not found.", "warning")
        return redirect(url_for('home'))
        
    # Verileri HTML sayfasına gönder / Send data to HTML page
    return render_template(
        'person_detail.html', 
        person=person_data,
        datetime_module=datetime, 
        quote_plus=quote_plus
    )


# =============================================================================
# Sistem Kontrolü / Health Check
# =============================================================================

@app.route('/health')
def health_check():
    """Sistem durumu kontrolü / System health check"""
    return jsonify(status="ok", timestamp=datetime.datetime.utcnow().isoformat()), 200


# =============================================================================
# Kimlik Doğrulama / Authentication
# =============================================================================

@app.route('/register', methods=['GET', 'POST'])
def register_route():
    """
    Kayıt sayfası / Registration page
    Yeni kullanıcıların sisteme kayıt olmasını sağlar.
    Allows new users to register to the system.
    """
    # 1. Kullanıcı zaten giriş yapmışsa ana sayfaya gönder / If user already logged in, send to home
    is_already_logged_in = current_user.is_authenticated
    if is_already_logged_in:
        return redirect(url_for('home'))

    # 2. Form gönderildiyse (POST isteği) / If form submitted (POST request)
    is_post_request = (request.method == 'POST')
    if is_post_request:
        # Formdaki kullanıcı adı ve şifreyi al / Get username and password from form
        username_input = request.form.get('username')
        password_input = request.form.get('password')

        # Girdileri kontrol et / Check inputs
        is_username_empty = not username_input
        is_password_empty = not password_input
        
        if is_username_empty or is_password_empty:
            flash('Kullanıcı adı ve şifre gereklidir. / Username and password are required.', 'danger')
        elif len(password_input) < 6:
            # Şifre çok kısa ise uyar / Warn if password is too short
            flash('Şifre en az 6 karakter olmalıdır. / Password must be at least 6 characters.', 'danger')
        else:
            # Kullanıcıyı veritabanında oluşturmaya çalış / Try to create user in database
            is_creation_successful = user_data.create_user(username_input, password_input)
            
            if is_creation_successful:
                # Başarılı kayıt sonrası giriş sayfasına yönlendir / Redirect to login after successful register
                flash('Hesabınız başarıyla oluşturuldu! Şimdi giriş yapabilirsiniz.', 'success')
                return redirect(url_for('login_route'))
            else:
                # Kullanıcı adı zaten varsa uyar / Warn if username already exists
                flash('Bu kullanıcı adı maalesef zaten kullanımda.', 'danger')

    # 3. Form gönderilmediyse (Sadece sayfayı açıyorsa) HTML sayfasını göster
    # 3. If form not submitted (Just opening the page), show HTML page
    return render_template(
        'register.html', 
        datetime_module=datetime,
        current_year=datetime.datetime.now().year
    )


@app.route('/login', methods=['GET', 'POST'])
def login_route():
    """
    Giriş sayfası / Login page
    Mevcut kullanıcıların sisteme giriş yapmasını sağlar.
    Allows existing users to login to the system.
    """
    # 1. Zaten giriş yapılmışsa ana sayfaya dön / If already logged in, go to home
    is_already_logged_in = current_user.is_authenticated
    if is_already_logged_in:
        return redirect(url_for('home'))

    # 2. Form gönderildiyse (POST) / If form submitted
    is_post_request = (request.method == 'POST')
    if is_post_request:
        # Formdan alınan bilgileri sakla / Store info from form
        username_input = request.form.get('username')
        password_input = request.form.get('password')

        # Veritabanında şifreyi doğrula / Verify password in database
        is_password_valid = user_data.check_user_password(username_input, password_input)
        
        if is_password_valid:
            # Şifre doğruysa Kullanıcı nesnesini (User object) oluştur
            # If password is correct, create User object
            user_obj = User.get(username_input)
            
            if user_obj is not None:
                # Kullanıcıyı sisteme aktif olarak dahil et (Oturum Aç)
                # Actively include user to the system (Log In)
                login_user(user_obj)
                flash('Başarıyla giriş yaptınız!', 'success')
                
                # Kullanıcı başka bir sayfadan yönlendirildiyse, oraya geri gönder
                # If user was redirected from another page, send them back there
                next_page = request.args.get('next')
                if next_page:
                    return redirect(next_page)
                else:
                    return redirect(url_for('home'))
            else:
                flash('Giriş sırasında sistem kaynaklı bir sorun oluştu.', 'danger')
        else:
            # Şifre veya isim yanlışsa uyar / Warn if password or name is wrong
            flash('Geçersiz kullanıcı adı veya şifre.', 'danger')

    # 3. Sayfayı normal gösterme (GET) / Normal page display (GET)
    return render_template(
        'login.html', 
        datetime_module=datetime,
        current_year=datetime.datetime.now().year
    )


@app.route('/logout')
@login_required
def logout_route():
    """Çıkış / Logout"""
    logout_user()
    flash('Başarıyla çıkış yaptınız.', 'info')
    return redirect(url_for('home'))


# =============================================================================
# Beğeni ve İzleme Listesi API / Like and Watchlist API
# =============================================================================

@app.route('/like_movie/<int:movie_id>', methods=['POST'])
@login_required
def like_movie(movie_id):
    """
    Filmi beğen/beğenmekten vazgeç / Like/unlike movie
    Kullanıcının bir filmi favori listesine eklemesini veya çıkarmasını sağlar.
    Allows user to add or remove a movie from their favorites list.
    """
    # 1. Kullanıcının tüm mevcut listelerini (beğeni, izleme vb.) dosyadan çek
    # 1. Fetch all existing lists (likes, watchlist etc.) of user from file
    user_interactions = user_data.load_user_interactions(current_user.username)
    liked_movies_list = user_interactions.get('liked_movies', [])

    # 2. Film numarasının (ID) listede olup olmadığını kontrol et
    # 2. Check if movie ID is in the list
    is_movie_already_liked = (movie_id in liked_movies_list)
    
    if is_movie_already_liked:
        # Zaten beğenilmişse, listeden çıkar / If already liked, remove from list
        liked_movies_list.remove(movie_id)
        action_performed = 'unliked'
    else:
        # Beğenilmemişse, listeye ekle / If not liked, add to list
        liked_movies_list.append(movie_id)
        action_performed = 'liked'

    # 3. Listeyi güncelleyip dosyaya tekrar kaydet / Update list and save back to file
    user_interactions['liked_movies'] = liked_movies_list
    user_data.save_user_interactions(current_user.username, user_interactions)
    
    # 4. İşlemin sonucunu çağrıyı yapan JavaScript koduna JSON olarak gönder
    # 4. Send action result as JSON to calling JavaScript code
    total_likes_count = len(liked_movies_list)
    return jsonify(
        status='success', 
        action=action_performed, 
        movie_id=movie_id,
        liked_count=total_likes_count
    )


@app.route('/watchlist_movie/<int:movie_id>', methods=['POST'])
@login_required
def watchlist_movie(movie_id):
    """
    İzleme listesine ekle/çıkar / Add/remove from watchlist
    Kullanıcının filmi izlenecekler listesine atmasını yönetir.
    Manages adding the movie to user's watchlist.
    """
    # 1. Kullanıcının tüm mevcut listelerini dosyadan çek
    # 1. Fetch all existing lists of user from file
    user_interactions = user_data.load_user_interactions(current_user.username)
    watchlist_movies_list = user_interactions.get('watchlist_movies', [])

    # 2. Film numarasının (ID) listede olup olmadığını kontrol et
    # 2. Check if movie ID is in the list
    is_movie_in_watchlist = (movie_id in watchlist_movies_list)
    
    if is_movie_in_watchlist:
        # Zaten listedeyse çıkar / If already in list, remove
        watchlist_movies_list.remove(movie_id)
        action_performed = 'removed'
    else:
        # Listede yoksa ekle / If not in list, add
        watchlist_movies_list.append(movie_id)
        action_performed = 'added'

    # 3. Listeyi güncelleyip dosyaya tekrar kaydet / Update list and save back to file
    user_interactions['watchlist_movies'] = watchlist_movies_list
    user_data.save_user_interactions(current_user.username, user_interactions)
    
    # 4. İşlemin sonucunu JSON olarak ekrana gönder / Send action result as JSON
    total_watchlist_count = len(watchlist_movies_list)
    return jsonify(
        status='success', 
        action=action_performed, 
        movie_id=movie_id,
        watchlist_count=total_watchlist_count
    )


# =============================================================================
# Kişisel Sayfa / Personal Page
# =============================================================================

@app.route('/my_lists')
@login_required
def personal_page_route():
    """
    Kişisel sayfa - beğeniler ve izleme listesi / Personal page - likes and watchlist
    Kullanıcının kaydettiği tüm filmleri derleyip HTML formatında sunar.
    Compiles all saved movies of the user and presents them in HTML format.
    """

    # 1. Başlangıç kontrollerini yap / Do startup checks
    is_engine_ready = (engine is not None)
    current_year_value = datetime.datetime.now().year
    active_username = current_user.username

    if not is_engine_ready:
        flash("Öneri motoru kullanılamıyor. Filmleriniz eksik görünebilir.", "danger")
        return render_template(
            'personal_page.html',
            liked_movies_details=[], 
            watchlist_movies_details=[],
            current_year=current_year_value,
            username=active_username
        )

    # 2. Kullanıcının seçtiği film ID'lerini dosyadan getir
    # 2. Get user's selected movie IDs from file
    user_interactions = user_data.load_user_interactions(active_username)
    liked_movie_ids = user_interactions.get('liked_movies', [])
    watchlist_movie_ids = user_interactions.get('watchlist_movies', [])

    # 3. Beğenilen filmlerin veritabanından isim/poster detaylarını çek
    # 3. Fetch name/poster details of liked movies from database
    liked_movies_full_details = []
    for movie_id in liked_movie_ids:
        # Film motorundan o filmin kısa bilgilerini talep et
        # Request short details of that movie from engine
        movie_short_detail = engine.get_movie_info_for_list(movie_id)
        
        # Eğer film silinmemişse ve duruyorsa listeye ekle
        # Add to list if movie is not deleted and exists
        if movie_short_detail is not None:
            liked_movies_full_details.append(movie_short_detail)

    # 4. İzleme listesindeki filmlerin detaylarını çek
    # 4. Fetch details of movies in watchlist
    watchlist_movies_full_details = []
    for movie_id in watchlist_movie_ids:
        movie_short_detail = engine.get_movie_info_for_list(movie_id)
        if movie_short_detail is not None:
            watchlist_movies_full_details.append(movie_short_detail)

    # 5. Filmleri alfabetik olarak (İsmine gör) sırala (Kullanıcı kolaylığı için)
    # 5. Sort movies alphabetically (By title) (For user convenience)
    
    # Lambda fonksiyonu, her film nesnesinden 'title' (başlık) özelliğini çeker
    # Lambda function extracts 'title' property from each film object
    def get_movie_title(movie_item):
        return movie_item.get('title', '')
        
    liked_movies_full_details.sort(key=get_movie_title)
    watchlist_movies_full_details.sort(key=get_movie_title)

    # 6. Tüm verileri toplayıp ekrana (HTML) aktar
    # 6. Collect all data and transfer to screen (HTML)
    return render_template(
        'personal_page.html',
        liked_movies_details=liked_movies_full_details,
        watchlist_movies_details=watchlist_movies_full_details,
        current_year=current_year_value,
        username=active_username
    )


# =============================================================================
# Film Arama / Movie Search
# =============================================================================

@app.route('/search_movie_redirect', methods=['POST'])
def search_movie_redirect_route():
    """Film arama ve yönlendirme / Movie search and redirect
    Kullanıcı arama çubuğundan film aratınca çalışır ve en yakın eşleşen filme yönlendirir.
    Runs when user searches a movie from navbar, redirects to best matching movie.
    """

    # 1. Öneri motoru çalışıyor mu kontrol et
    # 1. Check if engine is running
    is_engine_ready = (engine is not None)
    if not is_engine_ready or engine.movies_df.empty:
        flash("Öneri motoru kullanılamıyor.", "danger")
        return redirect(url_for('home'))

    # 2. Arama kutusundan gelen yazıyı al ve boşlukları temizle
    # 2. Get the search query and clean spaces
    raw_query = request.form.get('search_query', '')
    query = raw_query.strip()
    
    # Kullanıcı boş bir şey gönderdiyse geri ana sayfaya yolla
    # If query is empty, send back to home
    if not query:
        flash("Lütfen bir film adı girin.", "warning")
        
        previous_page = request.referrer
        if previous_page:
            return redirect(previous_page)
        else:
            return redirect(url_for('home'))

    # 3. Arama kelimesine en çok benzeyen (Fuzzy Match) filmi sistemde bul
    # 3. Find the movie most similar (Fuzzy Match) to the search term in the system
    all_titles_in_system = engine.all_movie_titles
    best_match_title = engine._find_best_match(query, all_titles_in_system)

    # 4. Eğer bir eşleşme (film) bulduysak, o filmin ID'sini bulup sayfasına yönlendir
    # 4. If we found a match (movie), find its ID and redirect to its page
    if best_match_title:
        # Tüm filmler tablosundan (Pandas Dataframe), başlığı bizim bulduğumuzla harfi harfine eşleşen satırı al
        # Get the row from movies table (Pandas Dataframe) where title exactly matches ours
        matched_rows_df = engine.movies_df[engine.movies_df['title'] == best_match_title]
        
        # Eğer tabloda bu satır varsa (Yani liste boş değilse)
        # If this row exists in table (Meaning list is not empty)
        is_movie_found_in_db = not matched_rows_df.empty
        if is_movie_found_in_db:
            try:
                # İlk eşleşen satırın 'id' sütunundaki numarayı tam sayı (int) olarak al
                # Get the number in 'id' column of first matched row as integer (int)
                first_matched_row = matched_rows_df.iloc[0]
                movie_target_id = int(first_matched_row['id'])
                
                # Her şey yolunda, kullanıcıyı bu kimlik numaralı (ID) detay sayfasına roketle
                # Everything is fine, rocket the user to the detail page with this ID
                return redirect(url_for('movie_details_page', movie_id=movie_target_id))
                
            except (ValueError, TypeError):
                # ID değerinde bir tip hatası (sayı değil de metin döndüyse)
                # If there's a type error in ID value (returned text instead of number)
                flash(f"'{best_match_title}' için geçerli bir kimlik numarası bulunamadı.", "danger")
        else:
            # Algoritma isim buldu ama veritabanında satır yok (Nadir durum)
            # Algorithm found name but no row in database (Rare edge case)
            flash(f"'{best_match_title}' adlı film listemizde bulunamadı.", "warning")
    else:
        # Hiçbir kelime bizim filmlerimizle %60 üstü bile olsa eşleşemedi
        # No word matched with our movies even over 60%
        flash(f"'{query}' ismine benzer hiçbir film bulamadık.", "warning")

    # Son olarak geldiği sayfaya (referrer) geri gönder (eğer yoksa ana sayfaya)
    # Finally, send back to previous page (or home if none)
    previous_page = request.referrer
    if previous_page:
        return redirect(previous_page)
    else:
        return redirect(url_for('home'))


# =============================================================================
# Uygulamayı Çalıştır / Run Application
# =============================================================================

if __name__ == '__main__':
    # Kullanıcıya uygulamanın nereden açılacağını net bir şekilde göster / Show user exactly where to open the app
    print("\n" + "=" * 60)
    print(" SISTEM HAZIR / SYSTEM READY ")
    print(" Tarayicinizda acin / Open in browser: http://localhost:5001")
    print(" Kapatmak icin / To exit: CTRL+C")
    print("=" * 60 + "\n")
    
    # use_reloader=False ekleyerek uygulamanın (ve öneri motorunun) iki kere yüklenmesini (double-load) engelledik.
    # Prevent double-loading of the app (and recommendation engine) by adding use_reloader=False.
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
