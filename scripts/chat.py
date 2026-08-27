"""Terminal sohbetinin giriş noktası.

Eğitilmiş bir checkpoint'i ve vocab.json'daki sözlüğü yükleyip modeli kurar,
ardından src/cli/chat.py içindeki sohbet döngüsünü başlatır.

Modelin kurulacağı boyutlar checkpoint'e gömülü config'den okunur; ağırlık
tensörlerinin şekli bu değerlere bağlı olduğu için elle verilmeleri hatalıdır.

    python scripts/chat.py
"""
