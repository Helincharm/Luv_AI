"""Model ve eğitim hiperparametrelerinin tek kaynağı.

ModelConfig modelin mimarisini tanımlar: katman sayısı, gömme boyutu,
attention başlığı sayısı ve bağlam uzunluğu. Bu değerler checkpoint ile
birlikte kaydedilir; eğitilmiş bir modeli geri yüklerken ağırlıkların şekli
bunlara bağlı olduğu için zorunludur.

TrainConfig yalnızca eğitim sırasında kullanılan ayarları tutar (öğrenme
oranı, batch boyutu, adım sayısı, cihaz). Çıkarım tarafında gerekmez, bu
yüzden mimariden ayrı tutulur.

Hazır ayarlar:
    ModelConfig.small()   -- boru hattını hızlıca doğrulamak için
    ModelConfig.medium()  -- gerçek eğitim için
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) n_head'e ({self.n_head}) tam bolunmeli"
            )

    @classmethod
    def small(cls, vocab_size: int) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            block_size=32,
            n_embd=64,
            n_head=4,
            n_layer=2,
            dropout=0.0,
        )

    @classmethod
    def medium(cls, vocab_size: int) -> "ModelConfig":
        return cls(
            vocab_size=vocab_size,
            block_size=256,
            n_embd=384,
            n_head=6,
            n_layer=6,
            dropout=0.2,
        )


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    batch_size: int = 16
    steps: int = 500
    eval_interval: int = 100
    device: str = "cpu"


# Manual test
# if __name__ == '__main__':
#     from dataclasses import asdict
#
#     vocab_size = 100
#
#     small_config = ModelConfig.small(vocab_size)
#     print(small_config)
#
#     medium_config = ModelConfig.medium(vocab_size)
#     print(medium_config)
#
#     config_dict = asdict(small_config)
#     print(config_dict)
