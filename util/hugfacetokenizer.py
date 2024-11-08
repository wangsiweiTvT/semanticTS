from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 初始化BPE模型
tokenizer = Tokenizer(models.BPE())

# 定义预处理器和训练器
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=1000, min_frequency=2)

# 训练模型
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# 保存模型
tokenizer.save("bpe-tokenizer.json")

# 加载模型
tokenizer = Tokenizer.from_file("bpe-tokenizer.json")

# 编码
encoded = tokenizer.encode("lower")
print(f"Encoded 'lower': {encoded.tokens}")

# 解码
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded text: {decoded}")
