"""Terminal sohbet döngüsü (REPL).

Kullanıcıdan girdi alır, tokenizer ile encode eder, generate() ile yanıt
üretir, decode edip ekrana yazar ve tekrar sorar.

Model ve tokenizer döngüye girmeden önce bir kez yüklenir; her turda yeniden
yüklemek checkpoint okumasını gereksizce tekrarlar ve yanıt gecikmesini
görünür biçimde artırır.
"""

import torch
from inference.generate import generate


def chat(model, tokenizer, block_size, max_new_tokens=100, temperature=0.8) -> None:
    print("Chat started. Type 'exit' to quit.")

    while True:
        try:
            input_text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if input_text == "exit":
            break

        if not input_text:
            continue

        unknown_chars = [
            char for char in input_text
            if char not in tokenizer.stoi
        ]

        if unknown_chars:
            print("Characters not in vocabulary:", "".join(sorted(set(unknown_chars))))
            continue

        token_ids = tokenizer.encode(input_text)
        input_tensor = torch.tensor(token_ids,
                                    dtype=torch.long).unsqueeze(0)

        output_ids = generate(
            model=model,
            idx=input_tensor,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            temperature=temperature
        )

        new_token_ids = output_ids[0, input_tensor.shape[1]:].tolist()
        response = tokenizer.decode(new_token_ids)
        print("ai  >", response)


# Manual test
# if __name__ == "__main__":
#     from data.tokenizer import CharTokenizer
#     from model.gpt import GPT
#
#     demo_tokenizer = CharTokenizer("the quick brown fox jumps over the lazy dog")
#
#     demo_model = GPT(vocab_size=demo_tokenizer.vocab_size,
#                      block_size=8,
#                      n_embd=16,
#                      n_head=4,
#                      n_layer=2)
#
#     chat(demo_model, demo_tokenizer, block_size=8, max_new_tokens=20)
