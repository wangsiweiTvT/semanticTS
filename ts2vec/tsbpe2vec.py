from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 使用 simple_preprocess 对每行文本进行预处理
            yield simple_preprocess(line)

# 文件路径
file_path = "../tokenize/BPEcrops"

# 读取并预处理语料
processed_sentences = list(read_corpus(file_path))


# 创建并训练 Word2Vec 模型
model = Word2Vec(sentences=processed_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 测试模型
word = "simple"
if word in model.wv:
    print(f"Vector for '{word}':", model.wv[word])

# 找到与某个词最相似的词
similar_words = model.wv.most_similar("simple", topn=5)
print("Most similar words to 'simple':", similar_words)