"""Terminal sohbet döngüsü (REPL).

Kullanıcıdan girdi alır, tokenizer ile encode eder, generate() ile yanıt
üretir, decode edip ekrana yazar ve tekrar sorar.

Model ve tokenizer döngüye girmeden önce bir kez yüklenir; her turda yeniden
yüklemek checkpoint okumasını gereksizce tekrarlar ve yanıt gecikmesini
görünür biçimde artırır.
"""
