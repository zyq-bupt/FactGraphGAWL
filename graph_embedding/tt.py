from transformers import PegasusTokenizer, PegasusModel

# 应该可以顺利加载
tokenizer = PegasusTokenizer.from_pretrained('/root/autodl-fs/zyq/pegasus-xsum')
model = PegasusModel.from_pretrained('/root/autodl-fs/zyq/pegasus-xsum')

print("✅ 句子片段（sentencepiece）已就绪，模型加载成功")
