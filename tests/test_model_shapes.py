"""GPT modeli şekil testleri.

Doğrulanan davranış: uçtan uca forward geçişinin (batch, seq_len) biçimli
token indekslerini alıp (batch, seq_len, vocab_size) biçimli logits ürettiği.
Şekil hataları eğitim sırasında anlaşılması güç mesajlarla ortaya çıktığı için
en ucuz güvenlik ağı budur.

    pytest tests/
"""
