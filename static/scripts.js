// Sayfa tamamen yüklendiğinde kodun çalışmasını sağlar
document.addEventListener('DOMContentLoaded', () => {

    // Elemanları güvenli bir şekilde almak için yardımcı fonksiyon
    const getElement = (selector, parent = document) => {
        const element = parent.querySelector(selector);
        if (!element) {
            // console.warn(`Uyarı: '${selector}' seçicisiyle eşleşen HTML elemanı bulunamadı.`);
        }
        return element;
    };
    const getAllElements = (selector, parent = document) => {
        const elements = parent.querySelectorAll(selector);
        if (elements.length === 0) {
            // console.warn(`Uyarı: '${selector}' seçicisiyle eşleşen HTML elemanları bulunamadı.`);
        }
        return elements;
    };

    // Toggle işlevselliğini kuran fonksiyon
    const setupToggle = (toggleSelector, selectWrapperSelector, inputWrapperSelector, selectInputSelector, customInputSelector) => {
        const toggle = getElement(toggleSelector);
        const selectWrapper = getElement(selectWrapperSelector); // Wrapper ID'leri güncellendi
        const inputWrapper = getElement(inputWrapperSelector);   // Wrapper ID'leri güncellendi
        const selectInput = getElement(selectInputSelector);
        const customInput = getElement(customInputSelector);

        if (!toggle || !selectWrapper || !inputWrapper || !selectInput || !customInput) {
            // console.warn(`Toggle kurulumu için gerekli elemanlar eksik: ${toggleSelector}`);
            return;
        }

        const updateVisibility = (isChecked) => {
            selectWrapper.classList.toggle('d-none', isChecked);
            selectWrapper.classList.toggle('d-block', !isChecked);
            inputWrapper.classList.toggle('d-block', isChecked);
            inputWrapper.classList.toggle('d-none', !isChecked);

            // Form gönderimi için name ve required attribute'larını ayarla
            // İsimler (name) input/select ID'leri ile aynı olmalı
            if (isChecked) {
                selectInput.name = ''; // Seçimden gelen veriyi gönderme
                selectInput.required = false;
                customInput.name = customInput.id; // Özel girişten gelen veriyi gönder
                // customInput.required = true; // Eğer özel giriş zorunluysa
            } else {
                selectInput.name = selectInput.id; // Seçimden gelen veriyi gönder
                // selectInput.required = true; // Eğer seçim zorunluysa
                customInput.name = ''; // Özel girişten gelen veriyi gönderme
                customInput.required = false;
            }
        };

        // Başlangıç durumu: HTML'de Jinja ile ayarlanan sınıflar ve form_data'daki 'checked' durumu esas alınır.
        // Toggle'ın mevcut durumuna göre ilk görünürlüğü ayarla.
        updateVisibility(toggle.checked); 

        toggle.addEventListener('change', () => {
            const isChecked = toggle.checked;
            updateVisibility(isChecked);
            if (isChecked) {
                customInput.focus();
            } else {
                // Eğer select input görünür hale geliyorsa ve bir değeri varsa, ona odaklanmak yerine
                // genel bir davranış sergileyebiliriz. Genellikle kullanıcı toggle'ı değiştirdiğinde
                // bir sonraki adıma kendi geçer.
            }
        });
    };

    // Film ve Yönetmen için Toggle mekanizmalarını başlat
    // Wrapper ID'leri HTML'deki gibi olmalı: #film_select_wrapper, #film_input_wrapper vb.
    setupToggle('#toggle_film', '#film_select_wrapper', '#film_input_wrapper', '#film_ismi', '#film_ismi_custom');
    setupToggle('#toggle_yonetmen', '#yonetmen_select_wrapper', '#yonetmen_input_wrapper', '#yonetmen', '#yonetmen_custom');


    // Form gönderildiğinde basit istemci taraflı doğrulama
    getAllElements('form').forEach(form => {
        form.addEventListener('submit', (event) => {
            let validationError = false;
            let firstErrorElement = null;

            // Hata mesajlarını temizle
            form.querySelectorAll('.error-message').forEach(el => el.remove());

            const createErrorMessage = (inputElement, message) => {
                const errorSpan = document.createElement('span');
                errorSpan.className = 'error-message';
                errorSpan.style.color = 'var(--danger-color)';
                errorSpan.style.fontSize = 'var(--font-size-sm)';
                errorSpan.style.display = 'block';
                errorSpan.style.marginTop = '0.25rem';
                errorSpan.textContent = message;
                inputElement.parentNode.insertBefore(errorSpan, inputElement.nextSibling);
                if (!firstErrorElement) firstErrorElement = inputElement;
                validationError = true;
            };

            // Öneri Sayısı Doğrulaması
            const numInput = getElement('input[name="num_recommendations"]', form);
            if (numInput) {
                const numValue = parseInt(numInput.value, 10);
                const min = parseInt(numInput.min, 10) || 1;
                const max = parseInt(numInput.max, 10) || MAX_RECOMMENDATION_COUNT; // MAX_RECOMMENDATION_COUNT JS'de tanımlı olmalı ya da HTML'den alınmalı

                if (numInput.value.trim() === '') {
                    // Boş bırakılabilir, backend varsayılanı kullanır.
                } else if (isNaN(numValue) || numValue < min || numValue > max) {
                    createErrorMessage(numInput, `Lütfen ${min} ile ${max} arasında bir sayı girin.`);
                }
            }

            // Yıl Aralığı Doğrulaması
            if (form.id === 'feature-form') {
                const minYearInput = getElement('#min_yil', form);
                const maxYearInput = getElement('#max_yil', form);
                if (minYearInput && maxYearInput) {
                    const minYear = minYearInput.value ? parseInt(minYearInput.value, 10) : null;
                    const maxYear = maxYearInput.value ? parseInt(maxYearInput.value, 10) : null;
                    const currentYear = new Date().getFullYear();

                    if (minYearInput.value && (isNaN(minYear) || minYear < 1900 || minYear > currentYear)) {
                         createErrorMessage(minYearInput, `Geçerli bir başlangıç yılı girin (1900 - ${currentYear}).`);
                    }
                    if (maxYearInput.value && (isNaN(maxYear) || maxYear < 1900 || maxYear > currentYear)) {
                         createErrorMessage(maxYearInput, `Geçerli bir bitiş yılı girin (1900 - ${currentYear}).`);
                    }
                    
                    if (minYear !== null && maxYear !== null && minYear > maxYear && !validationError) { // Sadece diğer yıl hataları yoksa bu hatayı göster
                        createErrorMessage(minYearInput, 'Başlangıç yılı, bitiş yılından büyük olamaz.');
                        // maxYearInput için de aynı mesajı ekleyebilir veya sadece birine odaklanabiliriz.
                    }
                }
            }
            
            // Benzerlik formu için film adı kontrolü (eğer toggle kapalıysa ve seçim yapılmadıysa)
            if (form.id === 'similarity-form') {
                const filmToggle = getElement('#toggle_film', form);
                const filmSelect = getElement('#film_ismi', form);
                const filmCustom = getElement('#film_ismi_custom', form);
                if (filmToggle && !filmToggle.checked && filmSelect && filmSelect.value === '') {
                    createErrorMessage(filmSelect, 'Lütfen popüler filmlerden birini seçin.');
                } else if (filmToggle && filmToggle.checked && filmCustom && filmCustom.value.trim() === '') {
                     createErrorMessage(filmCustom, 'Lütfen bir film adı yazın.');
                }
            }


            // Hata varsa formu gönderme ve odaklan
            if (validationError) {
                event.preventDefault();
                if (firstErrorElement) {
                    firstErrorElement.focus();
                    if (typeof firstErrorElement.select === 'function') {
                        firstErrorElement.select();
                    }
                }
            }
        });
    });
});
