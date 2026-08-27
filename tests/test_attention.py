"""Self-attention birim testleri.

Doğrulanan davranışlar: attention katmanının girdi/çıktı tensör şekillerinin
korunması ve nedensel maskenin gelecekteki tokenlara sızıntı yapmaması --
bir pozisyonun attention ağırlıklarında kendisinden sonraki sütunlar sıfır
olmalıdır.

    pytest tests/
"""
