"""Otoregresif metin üretimi.

Verilen bir başlangıç metninden itibaren token token yeni metin üretir: model
son konumun logits'ini verir, sampler bir token seçer, seçilen token girdinin
sonuna eklenir ve döngü tekrarlar.

Bağlam block_size'ı aştığında baştan kırpılır -- model, pozisyon gömme tablosu
kadar uzun bir diziden fazlasını göremez.
"""
