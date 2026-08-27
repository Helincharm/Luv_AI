"""Ana eğitim döngüsü.

Her adımda: batch al -> ileri geçiş -> kayıp -> gradyanları sıfırla -> geri
yayılım -> parametreleri güncelle.

zero_grad çağrısı zorunludur; PyTorch gradyanları varsayılan olarak
biriktirir, sıfırlanmazsa her adım önceki adımların gradyanlarını da taşır ve
güncellemeler bozulur.
"""

from training.dataset import get_batch
from training.loss import compute_loss


def train(model, optimizer, train_data, block_size, batch_size, steps):
    for step in range(steps):
        x, y= get_batch(train_data, block_size, batch_size)
        logits = model(x)
        loss = compute_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

# Manual test -- training loop
# if __name__ == "__main__":
#     import torch
#     from model.gpt import GPT
#     from training.optimizer import create_optimizer
#
#     demo_model = GPT(vocab_size=27,
#                         block_size=8,
#                         n_embd=16,
#                         n_head=4,
#                         n_layer=2)
#     demo_optimizer = create_optimizer(demo_model, learning_rate=0.001)
#     train_data = torch.load('../../data/processed/train.pt')
#     train(demo_model, demo_optimizer, train_data, block_size=8, batch_size=4, steps=500)


# Manual test -- checkpoint
# if __name__ == "__main__":
#     import torch
#     from model.gpt import GPT
#     from training.checkpoint import save_checkpoint
#     from training.optimizer import create_optimizer
#
#     demo_model = GPT(vocab_size=27,
#                          block_size=8,
#                          n_embd=16,
#                          n_head=4,
#                          n_layer=2)
#     demo_optimizer = create_optimizer(demo_model, learning_rate=0.001)
#     train_data = torch.load('../../data/processed/train.pt')
#     train(demo_model, demo_optimizer, train_data, block_size=8, batch_size=4, steps=500)
#     save_checkpoint(demo_model, demo_optimizer,"checkpoint.pth")
