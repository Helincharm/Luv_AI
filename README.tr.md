# Luv AI

[English](README.md) · **Türkçe**

PyTorch ile sıfırdan yazılan Transformer tabanlı bir dil modeli. Hazır model
kütüphanesi kullanılmadan; embedding katmanlarından self-attention'a, eğitim
döngüsünden metin üretimine kadar her parça adım adım elle yazılıyor.

Nihai hedef: web üzerinden sohbet edilebilen, Türkçe metin üreten bir model.

## Durum

Faz 1 (çekirdek mimari ve eğitim boru hattı) tamamlandı ve uçtan uca çalışıyor:
model eğitiliyor ve yeniden yüklenebilir bir checkpoint üretiyor. Sırada metin
üretimi ve sohbet arayüzü var.

```
step   0: loss 3.7816
step 100: loss 1.4456
step 200: loss 0.5093
step 300: loss 0.1970
step 400: loss 0.1355
```

Başlangıç kaybı `ln(vocab_size)` ile örtüşüyor — eğitilmemiş bir modelin alması
gereken değer bu. Şu anki örnek korpus çok küçük olduğu için sondaki düşük kayıp
ezberlemedir; bu aşamada beklenen davranış ve boru hattını doğrulamaya yeter.

## Yol Haritası

### Faz 1 — Çekirdek mimari ve eğitim boru hattı

- [x] Karakter seviyesinde tokenizer, encode/decode gidiş-dönüşü
- [x] Ham metinden train/val tensörlerine veri hazırlama
- [x] Token ve pozisyon gömme katmanları
- [x] Nedensel maskeli self-attention (tek ve çok başlıklı)
- [x] Pre-norm Transformer bloğu, artık bağlantılar
- [x] Uçtan uca GPT modeli
- [x] Cross-entropy kaybı ve geri yayılım
- [x] Rastgele pencereli batch üretimi
- [x] Eğitim döngüsü ve çalıştırılabilir giriş noktası
- [x] Checkpoint kaydetme ve geri yükleme
- [ ] Otoregresif metin üretimi *(`sampler.py` hazır, `generate.py` bekliyor)*
- [ ] Terminal sohbet arayüzü

### Faz 2 — Sağlamlaştırma

- [ ] `vocab.json` yüklenirken `itos` anahtarlarının tam sayıya dönmesi
- [ ] Checkpoint'e model konfigürasyonunun gömülmesi
- [ ] `torch.load` için `map_location` desteği (GPU'da eğitip CPU'da yükleme)
- [ ] Çıkarımda optimizer'sız checkpoint yükleme
- [ ] Doğrulama (validation) kaybının ölçülmesi
- [ ] `model.train()` / `model.eval()` mod yönetimi
- [ ] Cihaz (CPU/GPU) yönetimi
- [ ] Hiperparametre doğrulamaları ve kenar durum korumaları

### Faz 3 — Ölçekleme altyapısı

Büyük model eğitimi için zorunlu adımlar. Bunlar olmadan eğitim ya ıraksar ya
belleğe sığmaz.

- [ ] Öğrenme oranı zamanlayıcı (warmup + cosine decay)
- [ ] Gradyan kırpma (`clip_grad_norm_`)
- [ ] Karışık hassasiyet eğitimi (AMP / bf16)
- [ ] Gradyan biriktirme (küçük GPU'da büyük efektif batch)
- [ ] Bellek eşlemeli veri yükleme (`np.memmap`)
- [ ] Eğitimi checkpoint'ten kaldığı yerden sürdürme (adım sayacı dahil)
- [ ] Seçici weight decay (bias ve LayerNorm parametreleri hariç)
- [ ] En iyi doğrulama kaybına göre checkpoint saklama

### Faz 4 — Veri ve tokenizer

- [ ] Türkçe korpusun toplanması ve temizlenmesi
- [ ] BPE (subword) tokenizer — `src/data/bpe_tokenizer.py`
- [ ] Tokenizer eğitim betiği — `scripts/train_tokenizer.py`
- [ ] `tokenize_data.py` içinde tokenizer seçimi
- [ ] GPU üzerinde tam ölçekli eğitim

### Faz 5 — Model kalitesi ve hız

- [ ] Ağırlık paylaşımı (embedding ile lm_head aynı matrisi kullanır)
- [ ] Dropout'un modele bağlanması (config'de tanımlı, model henüz kullanmıyor)
- [ ] ReLU yerine GELU
- [ ] Ölçekli ağırlık başlatma (normal std=0.02, residual katmanlarda derinliğe göre)
- [ ] Birleşik QKV projeksiyonu (başlık başına döngü yerine tek matris)
- [ ] `F.scaled_dot_product_attention` (FlashAttention)
- [ ] Üretimde KV cache
- [ ] `torch.compile`
- [ ] RoPE — döndürmeli konum kodlaması

### Faz 6 — Değerlendirme

- [ ] Perplexity metriği
- [ ] Eğitim sırasında periyodik örnek üretimi
- [ ] Ayrı test kümesi (train / val / test)
- [ ] Sabit val batch'leri ile tekrar üretilebilir değerlendirme
- [ ] Verim ölçümü (token/saniye, MFU)

### Faz 7 — Sohbet davranışı

- [ ] Sohbet formatlı veri kümesi (kullanıcı / asistan yapısı)
- [ ] İnce ayar (fine-tuning) turu
- [ ] Üretimi durdurma mantığı
- [ ] Sistem yönergesi (system prompt) desteği

### Faz 8 — Web katmanı

- [ ] FastAPI sunucusu — `/chat` uç noktası
- [ ] Web arayüzü
- [ ] SSE ile token akışı
- [ ] Sohbet geçmişi yönetimi ve bağlam kırpma
- [ ] İstek gruplama (batching)
- [ ] Hız sınırlama, zaman aşımı, sağlık kontrolü uç noktası
- [ ] Yayına alma

### Faz 9 — Mühendislik olgunluğu

- [ ] `pyproject.toml` ile paketleme (`sys.path.insert` gereksinimini kaldırır)
- [ ] Betiklere `argparse` ile komut satırı argümanları
- [ ] Konfigürasyonu YAML/JSON dosyasından okuma, CLI ile ezme
- [ ] Deney takibi (TensorBoard veya Weights & Biases)
- [ ] Eğitim loglarının `reports/egitim.jsonl` dosyasına yazılması
- [ ] `ruff` + `black`, pre-commit hook
- [ ] Tüm modüller için pytest kapsamı
- [ ] GitHub Actions ile CI
- [x] Bağımlılık sürümlerinin sabitlenmesi
- [ ] Model kartı — parametre sayısı, veri, loss ve örnek çıktı tablosu

## En Yüksek Getirili Beş Adım

Faz 1 tamamlandıktan sonra sırasıyla:

1. **Karışık hassasiyet (AMP)** — iki-üç kat hız, yarı bellek
2. **LR zamanlayıcı + gradyan kırpma** — büyük modelde eğitim kararlılığının şartı
3. **KV cache** — üretimi 10-50 kat hızlandırır, sohbeti kullanılabilir kılar
4. **Ağırlık paylaşımı** — bedava parametre tasarrufu ve kalite artışı
5. **Periyodik örnek üretimi** — neyin işe yaradığını loss grafiğinden önce gösterir

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Kullanım

```bash
# Ham metni tokenize et, train/val/vocab dosyalarını üret
python scripts/tokenize_data.py
```

```bash
# Modeli eğit, checkpoints/luv_ai.pt üret
python scripts/train.py
```

<!-- Sohbet arayüzü tamamlanınca chat.py kullanımı buraya eklenecek. -->

## Proje Yapısı

```
config/     Model ve eğitim hiperparametreleri
data/       raw/ ham metin, processed/ tokenize edilmiş tensörler
scripts/    Çalıştırılabilir giriş noktaları
src/
    data/       Tokenizer ve veri hazırlama
    model/      Embedding, attention, transformer bloğu, GPT
    training/   Batch üretimi, kayıp, optimizer, checkpoint, eğitim döngüsü
    inference/  Örnekleme ve metin üretimi
    cli/        Terminal arayüzü
    utils/      Tohum sabitleme ve loglama
tests/      Birim testleri
```

## Mimari Notları

Yalnızca decoder'lı, GPT tarzı bir Transformer:

- Öğrenilebilir token ve pozisyon gömme katmanları
- Nedensel (causal) maskeli çok başlıklı self-attention
- Pre-norm düzeni ve artık (residual) bağlantılar
- `n_embd -> 4*n_embd -> n_embd` ileri beslemeli katman

### Konfigürasyonlar

| | `small` | `medium` | `large` *(planlanan)* |
|---|---|---|---|
| block_size | 32 | 256 | 512 |
| n_embd | 64 | 384 | 768 |
| n_head | 4 | 6 | 12 |
| n_layer | 2 | 6 | 12 |
| dropout | 0.0 | 0.2 | 0.1 |
| ~parametre | 0.1 M | 11 M | ~100 M |

`small` boru hattını hızlıca doğrulamak, `medium` ilk gerçek eğitim, `large`
nihai hedef içindir. `large`, GPT-2 small ile aynı ölçek sınıfındadır ve tek bir
16 GB GPU'da karışık hassasiyet ve gradyan biriktirme ile eğitilebilir; bunun
şartı Faz 3'ün tamamlanmış olmasıdır.

Şu anki tokenizer karakter seviyesindedir; Faz 4'te BPE'ye geçilecektir.

## Notlar

Bu, adım adım yazılan bir öğrenme projesidir ve hâlâ devam ediyor. Yukarıdaki
işaretsiz maddeler gerçekten yapılmamış işlerdir; kutucuklar deponun dürüst
durumunu gösterir.
