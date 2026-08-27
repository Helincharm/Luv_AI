"""Byte Pair Encoding (BPE) tokenizer.

Metni karakterlere değil, korpustan öğrenilmiş alt kelime (subword)
birimlerine böler. Sözlük sabit değildir; verinin üzerinde eğitilerek
oluşturulur: en sık geçen ardışık birim çifti tekrar tekrar birleştirilip yeni
bir token olarak sözlüğe eklenir.

Türkçe eklemeli bir dil olduğu için kazanç belirgindir -- "geliyorum" karakter
seviyesinde 9 token iken burada ~2 token olur ("gel" + "iyorum"). Model aynı
bağlam penceresinde birkaç kat daha fazla metin görür ve ekleri tek tek
ezberlemek yerine yeniden kullanılabilir birimler olarak öğrenir.

CharTokenizer ile aynı arayüzü sunar (encode / decode / vocab_size); böylece
model, eğitim ve çıkarım kodunun hiçbiri değişmeden ikisi arasında geçiş
yapılabilir. Değişen tek şey vocab_size'dır ve o da config üzerinden gelir.

CharTokenizer'dan temel farkı: bu tokenizer eğitilmek ve diske kaydedilmek
zorundadır. Çıkarım tarafı eğitimdekiyle birebir aynı birleştirme (merge)
listesini kullanmalıdır, aksi halde aynı sayı farklı metne çözülür.
"""
