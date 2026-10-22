# MovieRec

[![CI](https://github.com/alimustafaekmen/Movie-Recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/alimustafaekmen/Movie-Recommendation/actions/workflows/ci.yml)
[![Lisans: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)

[English](README.md) | **Türkçe**

**Bölümler:** [Ekran görüntüleri](#ekran-görüntüleri) · [Özellikler](#özellikler) · [Nasıl çalışır](#nasıl-çalışır) · [Değerlendirme](#değerlendirme) · [Yerelde çalıştırma](#yerelde-çalıştırma) · [Testler](#testler) · [Yapılandırma](#yapılandırma) · [Proje yapısı](#proje-yapısı) · [Sınırlamalar](#sınırlamalar) · [Lisans ve kaynaklar](#lisans-ve-kaynaklar)

MovieRec, Flask ile yazılmış içerik tabanlı bir film öneri sistemidir. [TMDB 5000 veri setindeki](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) 4.803 film üzerinde çalışır. Beğendiğiniz bir filmi verin, ona benzeyen filmleri bulsun. Tür, yönetmen, yıl ve puana göre de arama yapabilirsiniz. Arayüz Türkçedir.

**Canlı demo:** [movierec.pages.dev](https://movierec.pages.dev)

Demo, sonuçları önceden hesaplanmış statik bir kopyadır. Kişi sayfalarında biyografi yoktur ve yalnızca veri setindeki filmler listelenir.

## Ekran görüntüleri

<p>
  <a href="screenshots/home.jpg"><img src="screenshots/home.jpg" alt="Ana sayfa" width="49%"></a>
  <a href="screenshots/similar-movies.jpg"><img src="screenshots/similar-movies.jpg" alt="Inception için benzer filmler" width="49%"></a>
  <a href="screenshots/movie-page-dark.jpg"><img src="screenshots/movie-page-dark.jpg" alt="Koyu temada film sayfası" width="49%"></a>
  <a href="screenshots/my-lists.jpg"><img src="screenshots/my-lists.jpg" alt="Beğenilere göre öneriler" width="49%"></a>
</p>

## Özellikler

- **Benzer filmler:** Bir film adı girin, 12 benzer film gelsin. "Daha fazla göster" ile 30 filme kadar çıkar.
- **Filtreli arama:** Türlere, yönetmene, yayın yıllarına ve en düşük puana göre.
- **Film ve kişi sayfaları:** TMDb API'sinden posterler, Türkçe özetler, fragmanlar, oyuncular ve biyografiler.
- **Beğeniler ve izleme listesi:** Tarayıcınızda saklanır, bu yüzden üyelik ve veritabanı yoktur. Listeler sayfası ayrıca beğendiğiniz filmlere göre öneri yapar ("Beğendiklerinize göre").
- **Açık ve koyu tema:** Sistem ayarınızı izler. Üst menüdeki düğmeyle değiştirilebilir.

## Nasıl çalışır

### Benzer filmler

Her film dört TF-IDF bloğuna dönüşür. TF-IDF metni sayılara çevirir: bir terim, bir filmde sık ama tüm filmlerde nadir geçiyorsa yüksek puan alır.

| Blok | İçerik | Ağırlık | Benzerlikteki payı |
|---|---|---|---|
| Türler | tür adlarının tamamı ("Science Fiction" tek terimdir) | 0.3 | %14 |
| Yönetmenler | adların tamamı, her ortak yönetmen ayrı | 0.2 | %6 |
| Anahtar kelimeler | tek tek kelimeler, çünkü anahtar kelime öbeklerinin yarısından fazlası tek bir filme ait | 0.4 | %25 |
| Metin | özet ve slogan, "the" gibi yaygın İngilizce kelimeler çıkarılmış, en fazla 20.000 terim | 0.6 | %55 |

Her blok L2 ile normalize edilir (her filmin satırı 1 uzunluğuna ölçeklenir) ve ağırlığıyla çarpılır. Birleşen satır yeniden normalize edilir. Filmler sonra kosinüs benzerliğiyle karşılaştırılır. Bu ölçü, iki satırın ne kadar aynı yönü gösterdiğini 0 ile 1 arasında verir. Sonuç, dört bloğun benzerliklerinin ağırlıklı ortalamasıdır ve her blok ağırlığının karesi kadar sayılır. Tablonun son sütunu, dört alanı da dolu bir film için bu payları gösterir.

Ayrı bloklar, özetin sonuca ne kadar etki edebileceğini sınırlar. Önceki model bütün alanları tek bir metinde birleştiriyordu, bu yüzden nadir bir karakter adı ilgisiz filmleri birbirine bağlayabiliyordu. Toy Story'nin ilk 5 önerisinde *The 40 Year Old Virgin* vardı, çünkü iki filmin özetinde de "Andy" adı geçiyor.

- Önceki model: Toy Story 2, Toy Story 3, Monsters, Inc., Small Soldiers, The 40 Year Old Virgin
- Şimdi: Toy Story 3, Toy Story 2, Monsters, Inc., Big Hero 6, Inside Out

Model, yaklaşık 2,5 MB'lık seyrek bir 4.803 × 29.665 matristir. Seyrek, yalnızca sıfır olmayan değerlerin saklandığı anlamına gelir. Film film benzerlik tablosu saklanmaz. Bir filmin benzerlikleri, istek geldiğinde tek bir seyrek matris çarpımıyla hesaplanır.

### Yeniden sıralama

En benzer 91 film bir aday havuzu oluşturur. Bu sayı, en uzun listenin (30) üç katı artı birdir. Havuzdaki her film bir puan alır:

| Sinyal | Ölçekleme | Ağırlık |
|---|---|---|
| İçerik benzerliği | havuzdaki en iyi eşleşmeye bölünür | 0.50 |
| Oy ortalaması | 10'a bölünür | 0.20 |
| Popülerlik | log(1 + x), havuzdaki en büyük değere bölünür | 0.20 |
| Oy sayısı | log(1 + x), havuzdaki en büyük değere bölünür | 0.10 |

- Bir havuzdaki benzerlikler birbirine yakındır. En iyi eşleşmeye bölmek, 0 ile 1 arasındaki aralığın tamamını kullanır.
- Popülerliğin ortancası 13, en büyük değeri 876'dır. Logaritma olmasaydı tek bir gişe filmi, diğer bütün filmlerin popülerlik puanını sıfıra yaklaştırırdı. Oy sayıları da benzer durumdadır.
- TMDb puanları 100 oyun altında güvenilir değildir. Havuzda en az 100 oy almış bir film varsa bu filmler çıkarılır.
- Havuzun boyutu sabittir, bu yüzden daha fazla sonuç istemek ilk sonuçların sırasını değiştirmez.

### Ağırlıklar nasıl seçildi

Ağırlıklar elle seçilmiş yuvarlak sayılardır, çünkü ayarlamak için kullanıcı verisi yoktur. Tanınmış filmlerin listeleri okunarak ve `scripts/evaluate.py` sütunlarıyla kontrol edildiler.

- **İlk deneme (türler 0.4, yönetmenler 0.5, anahtar kelimeler 0.3):** 438 filmin 12 önerisinden 8'i ya da daha fazlası kendi yönetmeninden geliyordu. Jaws'un ilk 8 önerisinin hepsi Spielberg filmiydi.
- **Yönetmen 0.3'e indirildi:** 26 film. Ama aynı türden filmler, aynı temayı işleyen filmlerin önüne geçti. Seven Samurai'ye 1. ve 2. sırada boks filmleri geldi. Seri filmlerini bulma oranı %70,9'a düştü, önceki modelin %75,4'ünün altında kaldı.
- **Şimdi (türler 0.3, yönetmenler 0.2, anahtar kelimeler 0.4):** 1 film. Seven Samurai'nin boks filmleri 4. ve 5. sıraya iniyor. Seri filmlerini bulma oranı %75,1, önceki modele yakın.

Ağırlıklar [movierec/config.py](movierec/config.py) dosyasındadır.

### Beğendiklerinize göre

Listeler sayfası, beğendiğiniz ve izleme listenizdeki filmlerin numaralarını sunucuya gönderir. Sunucu, veri setinde bulunan en son eklenmiş 20 beğeniyi alır ve her biri için 30 benzer film bulur.

- Bir listede r. sıradaki film (sıfırdan sayarak) 1 / (r + 1) puan alır. Puanlar listeler boyunca toplanır, böylece birkaç beğeninin ortak filmleri yukarı çıkar.
- İki listeden birinde zaten olan filmler çıkarılır. Eşit puanlı filmler ilk göründükleri sırayı korur.
- Sayfa ilk 12 filmi gösterir. Sunucu hiçbir şey saklamaz.

### Filtreli arama

Arama; seçilen her türe, yönetmene, yıl ve puan sınırlarına uyan filmleri tutar. Birden çok yönetmeni olan bir film, yönetmenlerinden herhangi biriyle eşleşir. 100 oyun altındaki filmler, başka eşleşme yoksa dışarıda bırakılır. Kalanlar şu formülle sıralanır:

```text
0.40 * puan + 0.45 * popülerlik + 0.15 * oy sayısı
```

Puan 10'a bölünür. Popülerlik ve oy sayısı, eşleşen filmler arasındaki en büyük değere bölünür.

### Başlık araması

Büyük küçük harf fark etmeksizin tam eşleşen başlık kazanır. Aksi halde uygulama, yazım hatalarına izin veren bulanık eşleştirmeyle en yakın başlığı seçer. Bunun için thefuzz `WRatio` kullanılır ve 100 üzerinden en az 75 puan gerekir. Aranan ifadenin yarısından kısa bir başlık, yalnızca ifade o başlığın bütün kelimelerini içeriyorsa sayılır. Bu yüzden "Godfather 2" araması *Her* değil *The Godfather* filmini bulur. Aynı adı taşıyan birden çok film varsa en çok oy alan seçilir.

## Değerlendirme

Karşılaştırma yapılacak kullanıcı verisi yoktur. Bu yüzden `scripts/evaluate.py` her filmi sırayla başlangıç filmi olarak alır ve ona önerilen 12 filme bakar. Bilgileri başlangıç filmiyle uyuşuyor mu? Serisinin diğer filmlerini buluyor mu? Kataloğun ne kadarına ulaşıyor? Ne kadar popüler?

```bash
python scripts/evaluate.py
```

| Yöntem | Tür örtüşmesi | Aynı yönetmen | Ortak başrol oyuncusu | Seri filmlerini bulma | Katalog kapsamı | Ortanca oy |
|---|---|---|---|---|---|---|
| Saf kosinüs | 0.744 | %5,6 | %3,48 | %76,7 | %98,5 | 191 |
| MovieRec (hibrit) | 0.739 | %5,0 | %3,55 | %75,1 | %61,7 | 1.350 |
| En çok oy alanlar (taban çizgisi) | 0.159 | %0,1 | %0,75 | %1,1 | %0,3 | 10.099 |
| Önceki model, saf kosinüs | 0.356 | %4,5 | %3,23 | %76,4 | %98,6 | 284 |
| Önceki model, hibrit | 0.335 | %4,0 | %3,15 | %75,4 | %64,1 | 1.630 |

*Saf kosinüs*, yalnızca içerik benzerliğine göre ilk 12 filmdir. *MovieRec (hibrit)*, uygulamanın gösterdiği listedir. *En çok oy alanlar* taban çizgisi, her başlangıç filmine aynı 12 filmi verir. *Önceki model* (commit dc815a0) bütün alanlar için tek bir TF-IDF ve doğrusal popülerlik kullanıyordu. Satırları, aynı ölçüm kodunun o commit üzerinde çalıştırılmasıyla elde edildi. Betik yalnızca ilk üç satırı yazdırır.

- **Tür örtüşmesi:** başlangıç filminin ve önerilen filmin türleri arasındaki ortalama Jaccard indeksi. Jaccard, ortak türlerin iki filmin bütün türlerine bölümüdür. Türü olmayan başlangıç filmleri atlanır.
- **Aynı yönetmen:** başlangıç filmiyle yönetmeni ortak olan önerilerin oranı.
- **Ortak başrol oyuncusu:** ilk üç oyuncusundan biri başlangıç filminin ilk üç oyuncusundan biri olan önerilerin oranı. Oyuncular modelin girdisi değildir.
- **Seri filmlerini bulma:** başlangıç filminin seri filmlerinden 12 öneri içinde bulunanların oranı. Seri filmi, aynı kelimelerle başlayan farklı bir başlığa sahip ("Star Trek" gibi) ve en az 100 oy almış filmdir. Oran, böyle filmi olan 464 film üzerinden ortalanır. Yaklaşık her 10 başlık grubundan biri gerçek bir seri değildir, örneğin Kung Fu Hustle ile Kung Fu Panda.
- **Katalog kapsamı:** en az bir kez önerilen filmlerin bütün filmlere oranı.
- **Ortanca oy:** bütün önerilerin oy sayılarının ortancası. Bir film her önerildiğinde yeniden sayılır. Veri setinin ortancası 235'tir.

Sayıların gösterdikleri ve göstermedikleri:

- Doğruluğu değil, tutarlılığı ölçerler. Bir listenin kendi içinde anlamlı olup olmadığını gösterirler, insanların onu beğenip beğenmeyeceğini göstermezler.
- Türler ve yönetmenler modelin girdisidir, bu yüzden ilk iki sütun çoğunlukla modele verilen bilgiyi ölçer. Tür ağırlığı 0 yapıldığında hibritin tür örtüşmesi 0.312'dir. Yani 0.335'ten 0.739'a çıkış, büyük ölçüde modelin kendisine söyleneni yaptığını gösterir.
- Ortak başrol oyuncusu, yeni modelin daha iyi olduğunu göstermez. Yönetmenler oyuncularıyla tekrar çalışır, bu yüzden bu sütun yönetmen ağırlığıyla birlikte artar. Aynı yönetmenli çiftler çıkarıldığında bütün modeller yaklaşık %2,0 ile %2,2 arasında kalır. Seri filmlerini bulma da kısmen yapı gereğidir, çünkü devam filmlerinin özetlerinde aynı karakter adları geçer.
- Hibrit, biraz uyumdan ve yaklaşık 37 puanlık kapsamdan vazgeçip daha tanınmış filmler önerir (191'e karşı ortanca 1.350 oy). Logaritmik ölçekleme bu ortancayı bilerek düşürür. Doğrusal ölçeklemeyle 1.553 olurdu.
- Kapsam farkının büyük kısmı 100 oy sınırından gelir. Filmlerin yalnızca 3.162'sinin (%65,8) 100 ya da daha fazla oyu vardır ve hibrit bunların %94'üne ulaşır.

## Yerelde çalıştırma

Python 3.12 ve iki Kaggle CSV dosyası gerekir.

1. `tmdb_5000_movies.csv` ve `tmdb_5000_credits.csv` dosyalarını [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) üzerinden indirip `dataset/` klasörüne koyun.
2. `cp .env.example .env` komutunu çalıştırın ve iki değeri doldurun:
   - `TMDB_API_KEY`: TMDb API anahtarınız, [TMDb hesap ayarlarınızdan](https://www.themoviedb.org/settings/api). Daha uzun olan "API Read Access Token" değerini değil, "API Key" (v3) değerini kullanın. Anahtar olmadan uygulama yalnızca veri setiyle çalışır: poster, fragman, Türkçe özet ve kişi sayfası olmaz.
   - `FLASK_SECRET_KEY`: bilgi mesajlarını taşıyan çerezi imzalayan rastgele bir metin. `python -c "import secrets; print(secrets.token_hex(32))"` komutuyla üretebilirsiniz. Boş bırakılırsa uygulama her açılışta yeni bir tane üretir.
3. Kurun ve çalıştırın:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```

4. http://localhost:5001 adresini açın.

Geliştirme sunucusu yalnızca bu bilgisayardan erişilebilir (`HOST=127.0.0.1`). 5001 portunu kullanır, çünkü macOS 5000 portunu AirPlay Receiver için kullanır. Başka bir port için: `PORT=8000 python app.py`.

**Docker.** Önce CSV dosyalarını `dataset/` klasörüne koyun, çünkü imaj onları içine kopyalar. Sonra `docker compose up --build` komutu uygulamayı gunicorn ile http://localhost:5001 adresinde açar. Konteyner yetkisiz bir kullanıcıyla, sağlık kontrolüyle ve Linux yetkileri olmadan çalışır. Dosya sistemi, bellekteki `/tmp` dışında salt okunurdur ve yalnızca bu bilgisayardan erişilebilir.

## Testler

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest --cov
node --test "tests/js/*.test.mjs"
```

Testlerin koruduğu şeyler:

- **Motor çıktısı:** tam veri setinde başlık eşleşmelerinin ve benzer film listelerinin bir anlık görüntüsü (snapshot). Sıralamada bilinçli bir değişiklikten sonra `python tests/snapshot_engine.py --update` çalıştırılır ve fark incelenir.
- **TMDb hataları:** zaman aşımı, sunucu hatası ya da bozuk JSON durumunda sayfa yine veri setindeki bilgilerle açılır. Uygulama tek bir istek atar ve tekrar denemez. API anahtarı yoksa hiç istek atılmaz. Anahtar loglara asla yazılmaz.
- **Güvenlik:** her yanıtta İçerik Güvenliği Politikası (CSP) ve diğer başlıklar, istek boyutu sınırları ve site dışına çıkmayan yönlendirmeler.
- **Listeler ve değerlendirme:** tarayıcıdaki liste deposu ve listeler sayfası (`node:test` ile) ve `scripts/evaluate.py` içindeki ölçüm yardımcıları.

Kaggle CSV dosyaları yoksa pytest, `tests/fixtures/` içindeki 158 filmlik örneği kullanır ve tam veri seti gerektiren testleri atlar. CI her push ve pull request'te aynı komutları çalıştırır. `movierec/` ve `app.py` için satır ve dal kapsamı %95'in altına düşerse başarısız olur. Örnek veriyle kapsam yaklaşık %96'dır. CI ayrıca Docker imajını derler ve uygulamanın yetkisiz kullanıcıyla çalıştığını kontrol eder.

## Yapılandırma

Bu değerleri `.env` dosyasında (`.env.example` hepsini listeler) ya da ortam değişkenlerinde ayarlayın. İkisi birden varsa ortam değişkeni geçerli olur.

| Değişken | Görevi | Varsayılan |
|---|---|---|
| `TMDB_API_KEY` | TMDb API anahtarı (v3) | boş (yalnızca veri seti) |
| `FLASK_SECRET_KEY` | bilgi mesajı çerezini imzalar | her açılışta rastgele |
| `DATASET_DIR` | iki CSV dosyasının bulunduğu klasör | `dataset/` |
| `SESSION_COOKIE_SECURE` | HTTPS arkasında `1` yapılır: güvenli çerez ve HSTS | kapalı |
| `HOST`, `PORT`, `FLASK_DEBUG` | geliştirme sunucusu ayarları | `127.0.0.1`, `5001`, kapalı |
| `LOG_LEVEL` | log düzeyi, gunicorn altında da geçerli | `INFO` |

## Proje yapısı

```text
app.py                  giriş noktası (gunicorn app:app)
movierec/
  config.py             ortam değişkenlerinden ve .env dosyasından ayarlar
  dataset.py            CSV yükleme ve temizleme
  similarity.py         TF-IDF matrisi ve sıralama formülleri
  engine.py             RecommendationEngine: benzer filmler ve filtreli arama, ağ bağlantısı yok
  search_filters.py     girdi ayrıştırma, tür ve yönetmen filtreleri
  movie_details.py      TMDb verisiyle film sayfaları ve liste kartları
  genres.py             tür adları ve Türkçe karşılıkları
  tmdb.py               küçük bir TMDb API istemcisi
  people.py             TMDb'den yönetmen ve oyuncu sayfaları
  web/
    __init__.py         uygulama fabrikası: ayarlar, güvenlik başlıkları, hata sayfaları
    movies.py           ana sayfa, film, kişi ve arama rotaları, JSON önerileri
    lists.py            beğeniler ve izleme listesi sayfası, kart ve öneri uç noktaları
    redirects.py        önceki sayfaya güvenli yönlendirme
templates/              Jinja şablonları
static/                 CSS, JavaScript ve ikonlar. js/list_store.js listeleri localStorage içinde tutar
scripts/evaluate.py     benzer film listelerinin çevrimdışı değerlendirmesi
tests/                  pytest testleri. tests/js/ içinde node:test testleri var
```

## Sınırlamalar

- Veri seti 2017'de biter. Öneriler yalnızca bu 4.803 filmden gelir.
- Öneriler yalnızca içeriğe dayanır, kullanıcılardan öğrenmez.
- Bilgisi az olan filmler, çoğunlukla aynı türden filmlerin listesini alır. Casablanca'ya II. Dünya Savaşı filmleri yerine romantik filmler, Forrest Gump'a romantik komediler gelir.
- Bazı seriler yalnızca başlıklarıyla birbirine bağlıdır. Saw'un listesinde altı devam filminin hiçbiri yoktur.
- Tanınmış filmler bilerek öne çıkarılır, bu yüzden listeler kataloğun yalnızca yaklaşık %62'sine ulaşır.
- İngilizce olmayan filmlere nadiren kendi dillerinde öneri gelir. İngilizce olmayan 298 filmde önerilerin %2,9'u filmin özgün diliyle aynı dildedir.
- Filtreli arama, benzer filmlerden farklı olarak popülerliği ve oy sayısını hâlâ doğrusal ölçekler.
- Listeler JavaScript gerektirir. Tek bir cihazdaki tek bir tarayıcıya aittir ve site verileri silinince kaybolur.

## Lisans ve kaynaklar

Kaynak kod [PolyForm Noncommercial License 1.0.0](LICENSE) ile paylaşılmıştır. Kişisel, eğitim amaçlı ve diğer ticari olmayan amaçlarla kullanılabilir. Ticari kullanıma izin verilmez.

- Film verileri ve görseller [TMDB](https://www.themoviedb.org/) kaynaklıdır. Bu ürün TMDB API'sini kullanır ancak TMDB tarafından onaylanmış ya da sertifikalandırılmış değildir.
- Temel veri seti, Kaggle'daki [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) veri setidir.
- `static/icons.svg` içindeki ikonlar [Font Awesome Free 6.5.1](https://fontawesome.com) kaynaklıdır ve kendi [CC BY 4.0 lisanslarıyla](https://fontawesome.com/license/free) kullanılır.
- Yazı tipleri [IBM Plex Sans ve IBM Plex Sans Condensed](https://github.com/IBM/plex) olup SIL Open Font License 1.1 kapsamındadır ve Google Fonts üzerinden yüklenir.
