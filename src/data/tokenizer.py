"""Karakter seviyesinde tokenizer.

Metni modelin işleyebileceği tam sayı dizisine, üretilen tam sayıları da
tekrar metne çevirir. Sözlük eğitim korpusundan kurulur: metinde geçen
benzersiz karakterler sıralanıp her birine sabit bir indeks atanır.

stoi karakterden indekse, itos indeksten karaktere eşler. İkisi birbirinin
tersidir; encode/decode gidiş-dönüşü kayıpsızdır.
"""

class CharTokenizer:

    def __init__(self, text: str) -> None:
        unique_chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def from_stoi(cls, stoi: dict[str, int]) -> "CharTokenizer":
        tokenizer = cls("")
        tokenizer.stoi = dict(stoi)
        tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
        return tokenizer

# Manual test
# if __name__ == "__main__":
#     tokenizer = CharTokenizer("selam deneme")
#     print("stoi:", tokenizer.stoi)
#     print("itos:", tokenizer.itos)
#     print("vocab_size:", tokenizer.vocab_size)
#     print("encode('selam deneme'):", tokenizer.encode("selam deneme"))
