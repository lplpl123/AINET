import numpy as np


if __name__ == "__main__":
    # 加载数据集
    with open("./data/shakespeare.txt", encoding='utf-8') as text:
        text = text.read()
        # 文本中的非重复字符
        vocab = sorted(set(text))
        # 创建从非重复字符到索引的映射
        char2idx = {u:i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        text_as_int = np.array([char2idx[c] for c in text])
        # 设定每个句子长度的最大值
        seq_length = 100
        example_per_epoch = len(text)



