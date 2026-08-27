"""Rastgelelik kaynaklarının sabitlenmesi.

random, numpy ve torch birbirinden bağımsız rastgele sayı üreteçleri kullanır;
tekrar üretilebilir sonuç için üçünün de tohumlanması gerekir.

Sabit tohum olmadan iki koşu arasındaki farkın yaptığın değişiklikten mi yoksa
şanstan mı kaynaklandığı ayırt edilemez -- hata ayıklamayı mümkün kılan şey
budur. Eğitimin en başında, herhangi bir rastgele işlem yapılmadan çağrılır.
"""

import random
import torch
import numpy as np

def set_seed(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Manual test
# if __name__ == "__main__":
#     set_seed(42)
#     print (torch.randn(3))
