from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 创建一个空的 BPE Tokenizer
tokenizer = Tokenizer(BPE())

# 使用空格作为预分词器
tokenizer.pre_tokenizer = Whitespace()

# 创建一个 BPE 训练器
trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])

with open("BPEcrops", "r", encoding="utf-8") as f:
    lines = f.readlines()
# 训练 tokenizer
tokenizer.train_from_iterator(lines, trainer)
tokenizer.save("text-bpe-vocab.json")
# 测试分词
output = tokenizer.encode("While both language and time series are sequential in nature, they differ in terms of their representation — natural language consists of words from a finite vocabulary, while time series are real-valued.")
print("Tokens:", output.tokens)

# 打印词汇表
print("Vocabulary size:", tokenizer.get_vocab_size())
