"""BPE tokenizer'ının eğitilmesi ve diske kaydedilmesi.

Ham korpusu okur, BPETokenizer'ı hedef sözlük boyutuna kadar eğitir ve
öğrenilen birleştirme (merge) listesini data/processed/ altına yazar.

Model eğitiminden önce bir kez çalıştırılır. Sonuç dosyası hem
scripts/tokenize_data.py tarafından (veriyi encode etmek için) hem de çıkarım
tarafında (üretilen tokenları metne çevirmek için) kullanılır; ikisinin aynı
dosyayı okuması zorunludur.

    python scripts/train_tokenizer.py
"""
