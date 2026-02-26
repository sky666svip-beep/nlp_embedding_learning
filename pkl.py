import pickle

# 替换为你的.pkl文件路径
pkl_path = "output/tokenizer.pkl"

# 读取pkl文件中的分词器对象
with open(pkl_path, 'rb') as f:
    tokenizer = pickle.load(f)

# 打印对象类型（确认是分词器）
print("对象类型：", type(tokenizer))

# 核心：查看分词器的关键属性（根据NLP分词器的通用属性适配）
# 先尝试打印最常见的核心属性，覆盖90%的SimpleCharTokenizer场景
try:
    # 1. 词表（字符到ID的映射，核心！）
    if hasattr(tokenizer, 'vocab'):
        print("\n=== 分词器词表（前20个字符）===")
        vocab = tokenizer.vocab
        # 只显示前20个，避免内容太多
        for idx, (char, id_) in enumerate(vocab.items()):
            if idx < 20:
                print(f"字符: '{char}' → ID: {id_}")
            else:
                break

    # 2. 特殊字符（如PAD/UNK）
    if hasattr(tokenizer, 'pad_token'):
        print(f"\n=== 特殊字符 ===")
        print(f"填充符(PAD): {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"未知符(UNK): {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")

    # 3. 词表大小
    if hasattr(tokenizer, 'vocab_size'):
        print(f"\n=== 词表总大小 ===")
        print(f"分词器词表包含 {tokenizer.vocab_size} 个字符")

    # 4. 测试分词器功能（用一句话验证）
    test_sentence = "测试一下这个分词器"
    if hasattr(tokenizer, 'encode'):
        encoded = tokenizer.encode(test_sentence)
        print(f"\n=== 测试分词（句子：{test_sentence}）===")
        print(f"编码后的ID序列: {encoded}")
        # 反向解码（如果支持）
        if hasattr(tokenizer, 'decode'):
            decoded = tokenizer.decode(encoded)
            print(f"解码后的句子: {decoded}")

except Exception as e:
    print(f"\n⚠️  部分属性读取失败（不同分词器属性名可能不同）：{e}")
    print("\n=== 备选方案：查看对象所有属性 ===")
    # 打印该对象的所有属性名，你可以根据属性名针对性查看
    all_attrs = [attr for attr in dir(tokenizer) if not attr.startswith('__')]
    print("分词器对象的所有可访问属性：", all_attrs)