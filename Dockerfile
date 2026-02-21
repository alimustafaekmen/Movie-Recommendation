# Temel imaj olarak resmi Python imajını kullanıyoruz
# Use official Python image as a parent image
FROM python:3.10-slim

# Çalışma dizinini ayarlıyoruz
# Set the working directory in the container
WORKDIR /app

# İşletim sistemi bağımlılıklarını kur (thefuzz kütüphanesi derleme gerektirebilir)
# Install system dependencies (thefuzz library might require compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Önce sadece requirements.txt dosyasını kopyalıyoruz (Önbelleği verimli kullanmak için)
# Copy only requirements.txt first (To use cache efficiently)
COPY requirements.txt .

# Python bağımlılıklarını yüklüyoruz
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Uygulamanın geri kalanını (kodları) kopyalıyoruz
# Copy the rest of the application (codes)
COPY . .

# Flask'ın internete açılacağı portu belirtiyoruz
# Expose the port that Flask will run on
EXPOSE 5001

# Konteyner ayağa kalktığında çalıştırılacak komut
# Command to run when the container starts
CMD ["python", "app.py"]
