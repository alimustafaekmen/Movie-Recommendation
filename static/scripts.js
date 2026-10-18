// Behaviour shared by every page; the home page forms need most of it.
document.addEventListener('DOMContentLoaded', () => {

    // Flash dismiss, back links and list toggles are delegated, so they also work on content added later.
    document.addEventListener('click', (event) => {
        const dismiss = event.target.closest('[data-dismiss="alert"]');
        if (dismiss) {
            dismiss.closest('.alert').remove();
        }

        // Go back only when the previous page is ours; the link's own href (home) is the fallback.
        const back = event.target.closest('[data-back]');
        if (back && document.referrer.startsWith(location.origin + '/')) {
            event.preventDefault();
            history.back();
        }

        const toggle = event.target.closest('[aria-controls][aria-expanded]');
        if (toggle) {
            const expanded = toggle.getAttribute('aria-expanded') === 'true';
            toggle.setAttribute('aria-expanded', String(!expanded));
            document.getElementById(toggle.getAttribute('aria-controls')).hidden = expanded;
        }
    });

    // The suggestion lists are large, so they are fetched when the user first focuses the input.
    document.querySelectorAll('datalist[data-src]').forEach((list) => {
        const input = document.querySelector(`input[list="${list.id}"]`);
        if (!input) {
            return;
        }
        input.addEventListener('focus', async () => {
            try {
                const response = await fetch(list.dataset.src);
                const names = response.ok ? await response.json() : [];
                list.replaceChildren(...names.map((name) => new Option(name)));
            } catch {
                // The suggestions are optional; typing still works without them.
            }
        }, { once: true });
    });

    // Show either the select or the free-text input, and submit only the visible one.
    const setupToggle = (toggleSelector, selectWrapperSelector, inputWrapperSelector, selectInputSelector, customInputSelector) => {
        const toggle = document.querySelector(toggleSelector);
        const selectWrapper = document.querySelector(selectWrapperSelector);
        const inputWrapper = document.querySelector(inputWrapperSelector);
        const selectInput = document.querySelector(selectInputSelector);
        const customInput = document.querySelector(customInputSelector);

        if (!toggle || !selectWrapper || !inputWrapper || !selectInput || !customInput) {
            return;
        }

        const updateVisibility = (isChecked) => {
            selectWrapper.classList.toggle('d-none', isChecked);
            selectWrapper.classList.toggle('d-block', !isChecked);
            inputWrapper.classList.toggle('d-block', isChecked);
            inputWrapper.classList.toggle('d-none', !isChecked);

            // The server reads the field named after the input's id; the hidden one gets no name.
            if (isChecked) {
                selectInput.name = '';
                selectInput.required = false;
                customInput.name = customInput.id;
            } else {
                selectInput.name = selectInput.id;
                customInput.name = '';
                customInput.required = false;
            }
        };

        updateVisibility(toggle.checked);

        toggle.addEventListener('change', () => {
            updateVisibility(toggle.checked);
            if (toggle.checked) {
                customInput.focus();
            }
        });
    };

    setupToggle('#toggle_film', '#film_select_wrapper', '#film_input_wrapper', '#film_ismi', '#film_ismi_custom');
    setupToggle('#toggle_yonetmen', '#yonetmen_select_wrapper', '#yonetmen_input_wrapper', '#yonetmen', '#yonetmen_custom');


    // Basic client-side validation before a form is submitted.
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', (event) => {
            let validationError = false;
            let firstErrorElement = null;

            form.querySelectorAll('.error-message').forEach(el => el.remove());

            const createErrorMessage = (inputElement, message) => {
                const errorSpan = document.createElement('span');
                errorSpan.className = 'error-message';
                errorSpan.textContent = message;
                inputElement.parentNode.insertBefore(errorSpan, inputElement.nextSibling);
                if (!firstErrorElement) firstErrorElement = inputElement;
                validationError = true;
            };

            const numInput = form.querySelector('input[name="num_recommendations"]');
            if (numInput) {
                const numValue = parseInt(numInput.value, 10);
                const min = parseInt(numInput.min, 10) || 1;
                const max = parseInt(numInput.max, 10);

                // An empty count is fine: the server uses its default.
                if (numInput.value.trim() !== '' && (isNaN(numValue) || numValue < min || numValue > max)) {
                    createErrorMessage(numInput, `Lütfen ${min} ile ${max} arasında bir sayı girin.`);
                }
            }

            if (form.id === 'feature-form') {
                const minYearInput = form.querySelector('#min_yil');
                const maxYearInput = form.querySelector('#max_yil');
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

                    // Only report the range when both years are valid on their own.
                    if (minYear !== null && maxYear !== null && minYear > maxYear && !validationError) {
                        createErrorMessage(minYearInput, 'Başlangıç yılı, bitiş yılından büyük olamaz.');
                    }
                }
            }

            if (form.id === 'similarity-form') {
                const filmToggle = form.querySelector('#toggle_film');
                const filmSelect = form.querySelector('#film_ismi');
                const filmCustom = form.querySelector('#film_ismi_custom');
                if (filmToggle && !filmToggle.checked && filmSelect && filmSelect.value === '') {
                    createErrorMessage(filmSelect, 'Lütfen popüler filmlerden birini seçin.');
                } else if (filmToggle && filmToggle.checked && filmCustom && filmCustom.value.trim() === '') {
                    createErrorMessage(filmCustom, 'Lütfen bir film adı yazın.');
                }
            }

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
