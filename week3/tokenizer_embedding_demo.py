import torch
from torch import nn


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

#创建 Tokenizer 对象
# 收集所有字符
# → 去重和排序
# → 加入[PAD]、[UNK]
# → 建立token_to_id
# → 建立id_to_token
class SimpleCharTokenizer: #定义字符级Tokenizer类，它负责保存词表并提供编码能力

    def __init__(self, texts):
        all_characters = "".join(texts) #把多段文本拼成一个字符串，用于统计出现过的字符；"".join(texts)不加分隔符；",".join(texts)使用逗号连接
        unique_characters = sorted(set(all_characters)) #set()去重;sorted()按照字符的Unicode编码值排序，使每次运行生成相同词表
        #如果不排序，集合遍历顺序可能导致Token ID不稳定

        #先把特殊Token放进词表，并固定ID
        #[PAD]把不同长度序列补到相同长度
        #[UNK]用于表示词表不存在的真实字符
        #注意二者不同：
        #[PAD]：这里没有真实内容，Mask为0。方便初始化、Padding和Mask处理。
        #[UNK]：这里有真实内容，只是词表不认识，Mask仍为1。
        self.token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }

        for character in unique_characters:
            self.token_to_id[character] = len(
                self.token_to_id
            ) #当前词表长度就是下一个可用ID，最终每个Token都有唯一ID

        #字典推导式
        #编码：token_to_id，文字变ID
        #解码：id_to_token，ID变回文字
        self.id_to_token = { #推导式交换它们的位置
            token_id: token  
            for token, token_id in self.token_to_id.items()
        }

        self.pad_token_id = self.token_to_id[PAD_TOKEN] #Padding使用哪个ID
        self.unk_token_id = self.token_to_id[UNK_TOKEN] #未知字符使用哪个ID
        self.vocab_size = len(self.token_to_id) #词表总行数

    def encode(self, text, max_length): #把一条字符串编码成固定长度的token_ids和attention_mask
        token_ids = []

        # 字典的 get(key, default)：
        # 找到字符：返回真实ID。
        # 找不到字符：返回 [UNK] 的ID。
        for character in text:
            token_id = self.token_to_id.get(
                character,
                self.unk_token_id,
            )
            token_ids.append(token_id)

        token_ids = token_ids[:max_length] #截断，只保留前 max_length 个Token

        attention_mask = [1] * len(token_ids) #如果截断后有5个Token：[1] * 5，得到：[1, 1, 1, 1, 1]，这里的1表示真实Token，包括 [UNK]

        padding_length = max_length - len(token_ids)

        token_ids += [ #+=表示把右侧列表元素追加到原列表
            self.pad_token_id
        ] * padding_length

        attention_mask += [0] * padding_length #给Mask补0，Token ID和Mask必须逐位置对应

        return token_ids, attention_mask #这是返回一个二元组，(token_ids, attention_mask)
        #调用方可以解包：token_ids, attention_mask = tokenizer.encode(...)


def build_batch(
    tokenizer, #已经建立好词表的分词器对象
    texts, #多条原始文本
    max_length, #每条文本统一到多长
):
    batch_token_ids = []
    batch_attention_masks = []

    #逐条编码
    for text in texts:
        #encode() 返回一个二元组：(token_ids, attention_mask)，使用解包赋值
        token_ids, attention_mask = tokenizer.encode(
            text=text,
            max_length=max_length,
        )

        #收集整批数据
        batch_token_ids.append(token_ids)
        batch_attention_masks.append(attention_mask)

    #转成 PyTorch Tensor
    token_ids_tensor = torch.tensor(
        batch_token_ids,
        dtype=torch.long, #dtype = torch.int64，也就是 torch.long
    )
    #为什么必须使用 torch.long？因为 Token ID 是整数索引，nn.Embedding 要用这些整数定位词向量表中的某一行。

    attention_mask_tensor = torch.tensor(
        batch_attention_masks,
        dtype=torch.long,
    )

    return token_ids_tensor, attention_mask_tensor


def mean_pool(
    token_embeddings,
    attention_mask,
):
    expanded_mask = attention_mask.unsqueeze(
        dim=-1 #dim=-1 表示在最后增加一个维度，方便后面广播
    ).to(token_embeddings.dtype) #把 Mask 转换成与 Embedding 相同的类型：int64 → float32

    masked_embeddings = (
        token_embeddings * expanded_mask
    )
    # 对于真实 Token：Token向量 × 1 = 原Token向量
    # 对于 Padding：：Token向量 × 0 = 全零向量

    # masked_embeddings 有三个维度：
    # dim=0：Batch维度，3条文本
    # dim=1：序列维度，每条12个Token
    # dim=2：特征维度，每个Token有8个数字
    summed_embeddings = masked_embeddings.sum(
        dim=1 #沿着序列维度求和
    )
    #不是对 dim=2 求和，因为 dim=2 是向量内部的特征维度，求和后会破坏向量表示

    valid_token_counts = expanded_mask.sum( #计算每条文本的有效 Token 数量
        dim=1
    ).clamp(min=1.0) #表示最小值不能低于 1，用于防止一条文本全是 Padding 时出现除以零

    text_embeddings = (
        summed_embeddings / valid_token_counts #计算平均文本向量：有效Token向量之和 / 有效Token数量
    )

    return text_embeddings

# 准备文本
# → 建立Tokenizer
# → 批量编码
# → 建立Embedding层
# → 查表得到Token向量
# → 池化得到文本向量
# → 打印结果
def main():
    torch.manual_seed(42) #固定随机数

    # 准备文本
    texts = [ #Batch Size = 3
        "智能座舱支持语音控制",
        "语音助手可以控制空调",
        "车辆支持导航",
    ]

    # → 建立Tokenizer
    tokenizer = SimpleCharTokenizer(texts)
    #tokenizer保存的是对象，不是编码后的结果，这个对象内部保存了词表和 encode() 方法

    # → 批量编码
    token_ids, attention_mask = build_batch(
        #等号左边是函数形参，右边是当前变量或具体数值
        tokenizer=tokenizer,
        texts=texts,
        max_length=12,
    )
    # token_ids.shape      = [3, 12]
    # attention_mask.shape = [3, 12]

    # → 建立Embedding层
    embedding_layer = nn.Embedding( #nn.Embedding 本质上是一张可以训练的向量表
        num_embeddings=tokenizer.vocab_size, #当前的词表大小，这里是22，表示表里有多少行，即可以识别多少个 Token ID
        embedding_dim=8, #向量维度，表示每个 Token 用多少个数字表示；真实 Embedding 模型可能使用 384、768、1024 等维度
        padding_idx=tokenizer.pad_token_id, #指定[PAD]对应的行，PyTorch会让这一行保持为零，并且训练时通常不更新该行，从而避免填充位置学习出实际含义。
    )
    # embedding_layer.weight.shape = [22, 8]
    # Token ID 就是这张表的行号：
    # token_id = 5
    # → 取出 weight[5]
    # → 得到长度为8的向量

    with torch.no_grad():
        # → 查表得到Token向量
        token_embeddings = embedding_layer(
            token_ids
        )
        #token_ids.shape = [3, 12]；dtype = int64
        #每个整数 ID 都会从 Embedding 表中查出一个长度为 8 的向量
        #token_embeddings.shape = [3, 12, 8]；dtype = float32
        #需要注意：当前这些向量是随机初始化的，没有经过训练，因此还不具备真正的语义表达能力。

        # → 池化得到文本向量 
        # mean_pool 为什么需要 Mask？目标是把token_embeddings: [B, S, H] 变成：text_embeddings: [B, H]
        # 也就是把一条文本里的多个 Token 向量合并成一个文本向量
        text_embeddings = mean_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        #token_embeddings = [3, 12, 8]；attention_mask = [3, 12]
        # 扩展Mask       [3, 12, 1]
        # 屏蔽Padding    [3, 12, 8]
        # Token求和      [3, 8]
        # 有效数量       [3, 1]
        # 平均池化       [3, 8]

    #查看词表和词表大小
    print("vocab:", tokenizer.token_to_id) #vocab：按照 Unicode 排序建立 Token 和 ID 的映射
    print("vocab size:", tokenizer.vocab_size)

    print("\ntoken ids:")
    print(token_ids) #token_ids 的行顺序：按照 texts 中三条文本的顺序

    print("\nattention mask:")
    print(attention_mask)

    print(
        "\ntoken embeddings shape:",
        token_embeddings.shape,
    )

    print(
        "text embeddings shape:",
        text_embeddings.shape,
    )


if __name__ == "__main__":
    main()