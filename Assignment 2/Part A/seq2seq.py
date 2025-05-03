#%%
import re

# 修改normalize函数，处理中英文差异
def normalizeEnglish(s):
    s = s.strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.lower()

def normalizeChinese(s):
    s = s.strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"\s+", " ", s)
    return s

# 修改Lang类，支持按字符分割中文
class Lang:
    def __init__(self, name, is_chinese=False):
        self.name = name
        self.is_chinese = is_chinese
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.word2count = {"SOS": 1, "EOS": 1, "UNK": 1}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3

    def addSentence(self, sentence):
        if self.is_chinese:
            for char in sentence:
                self.addWord(char)
        else:
            for word in sentence.split():
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# 读取数据集并划分训练测试
def readLangs(lang1, lang2, reverse=False):
    lines = open('./cmn-eng/cmn.txt', encoding='utf-8').read().strip().split('\n')
    pairs = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            eng = normalizeEnglish(parts[0])
            chn = normalizeChinese(parts[1])
            pairs.append([chn, eng] if reverse else [eng, chn])
    input_lang = Lang(lang1, is_chinese=(lang1 == 'chn'))
    output_lang = Lang(lang2, is_chinese=(lang2 == 'chn'))
    return input_lang, output_lang, pairs

# 划分数据集并构建词汇表
from sklearn.model_selection import train_test_split

input_lang, output_lang, pairs = readLangs('chn', 'eng', reverse=True)
train_pairs, test_pairs = train_test_split(pairs, test_size=0.1, random_state=42)

# 仅用训练集构建词汇表
for pair in train_pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

MAX_LENGTH = 35  # 根据数据调整

def indexesFromSentence(lang, sentence):
    indexes = []
    if lang.is_chinese:
        for char in sentence:
            indexes.append(lang.word2index.get(char, 2))  # UNK=2
    else:
        for word in sentence.split():
            indexes.append(lang.word2index.get(word, 2))
    indexes.append(1)  # EOS
    return indexes

def tensorFromPair(input_lang, output_lang, pair):
    input_ids = indexesFromSentence(input_lang, pair[0])
    target_ids = indexesFromSentence(output_lang, pair[1])
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device).view(-1, 1),
        torch.tensor(target_ids, dtype=torch.long, device=device).view(-1, 1)
    )

# 创建DataLoader
from torch.utils.data import DataLoader, TensorDataset

def prepareDataLoader(pairs, input_lang, output_lang, batch_size=32):
    input_ids = []
    target_ids = []
    for pair in pairs:
        input_tensor, target_tensor = tensorFromPair(input_lang, output_lang, pair)
        input_ids.append(input_tensor.squeeze())
        target_ids.append(target_tensor.squeeze())
    
    # 填充序列至MAX_LENGTH
    input_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    dataset = TensorDataset(input_padded, target_padded)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = prepareDataLoader(train_pairs, input_lang, output_lang)
test_loader = prepareDataLoader(test_pairs, input_lang, output_lang)

# %%
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.attention = BahdanauAttention(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        # 1. 嵌入当前输入
        embedded = self.embedding(decoder_input).unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # 2. 计算注意力权重并应用到编码器输出
        # 确保 query 的维度为 (batch_size, 1, hidden_size)
        query = decoder_hidden.permute(1, 0, 2)  # GRU 的 hidden 形状为 (num_layers, batch, hidden_size)
        context, weights = self.attention(query, encoder_outputs)  # (batch_size, 1, hidden_size)
        
        # 3. 将嵌入和上下文拼接后输入 GRU
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, 2*hidden_size)
        output, hidden = self.gru(rnn_input, decoder_hidden)  # hidden 形状需与 GRU 层数匹配
        
        # 4. 生成输出
        output = self.out(output.squeeze(1))  # (batch_size, output_size)
        return output, hidden, weights.squeeze(1)  # (batch_size, seq_len, 1)

    def forward(self, encoder_outputs, encoder_hidden, target=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.tensor([SOS_token] * batch_size, device=device)  # (batch_size,)
        decoder_hidden = encoder_hidden
        outputs = []
        attentions = []
        
        for _ in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attention = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs.append(decoder_output)
            attentions.append(attention)
            decoder_input = decoder_output.argmax(-1)  # (batch_size,)
        return torch.stack(outputs, dim=1), decoder_hidden, torch.stack(attentions, dim=1)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        # query: (batch_size, 1, hidden_size)
        # keys: (batch_size, seq_len, hidden_size)
        scores = self.V(torch.tanh(self.W(query) + keys))  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.bmm(weights.transpose(1, 2), keys)  # (batch_size, 1, hidden_size)
        return context, weights  # 返回上下文和注意力权重

#%%
import torch.optim as optim
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')  # 使用更好看的样式

SOS_token = 0
EOS_token = 1

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# 在训练循环中添加进度条
from tqdm import tqdm

def train(dataloader, epochs=10):
    encoder = EncoderRNN(input_lang.n_words, 256).to(device)
    decoder = AttnDecoderRNN(256, output_lang.n_words).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            enc_outputs, enc_hidden = encoder(inputs)
            
            # 修正点：接收三个返回值
            dec_outputs, _, _ = decoder(enc_outputs, enc_hidden)  # 新增第三个占位符 _
            
            # 截断输出以匹配目标长度
            dec_outputs = dec_outputs[:, :targets.size(1), :]
            loss = criterion(dec_outputs.reshape(-1, dec_outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    # ...保存模型和绘图代码...
    
    # 新增：绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, epochs, 2), range(1, epochs+1, 2))  
    plt.legend()
    plt.savefig('training_loss.png')  # 保存为图片
    plt.show()
    
    # 保存模型权重
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
    return train_losses  # 返回损失列表供后续分析

def evaluate(sentence):
    encoder = EncoderRNN(input_lang.n_words, 256).to(device)
    decoder = AttnDecoderRNN(256, output_lang.n_words).to(device)
    
    # 加载训练好的权重
    encoder.load_state_dict(torch.load("encoder.pth"))
    decoder.load_state_dict(torch.load("decoder.pth"))
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        enc_outputs, enc_hidden = encoder(input_tensor)
        dec_outputs, _, _ = decoder(enc_outputs, enc_hidden)  # 同样接收三个值
        decoded_ids = dec_outputs.argmax(-1).squeeze().tolist()
        return ' '.join([output_lang.index2word[idx] for idx in decoded_ids if idx not in (0, 1)])

# %%
# 在训练前检查模型参数是否可训练

train(train_loader, epochs=20)
# 测试模型  
test_sentence = "香港中文大学深圳世界一流。"
translated = evaluate(test_sentence)
print(f"Input: {test_sentence}")
print(f"Translated: {translated}")


# %%
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_attention(input_sentence, output_words, attentions):
    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
    attentions = attentions.cpu().numpy()
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # 显示注意力权重矩阵
    cax = ax.matshow(attentions, cmap='viridis')
    fig.colorbar(cax)

    # 设置坐标标签
    ax.set_xticks(range(len(input_sentence.split()) + 2))  # 确保刻度数量与标签数量一致
    ax.set_xticklabels([''] + input_sentence.split() + ['<EOS>'], rotation=90)
    ax.set_yticks(range(len(output_words) + 1))
    ax.set_yticklabels([''] + output_words)

    plt.show()

def enhanced_evaluate(sentence, show_attention=False):
    encoder = EncoderRNN(input_lang.n_words, 256).to(device)
    decoder = AttnDecoderRNN(256, output_lang.n_words).to(device)
    
    # 加载训练好的权重
    encoder.load_state_dict(torch.load("encoder.pth"))
    decoder.load_state_dict(torch.load("decoder.pth"))

    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        enc_outputs, enc_hidden = encoder(input_tensor)
        dec_outputs, _, attentions = decoder(enc_outputs, enc_hidden)
        
        decoded_ids = dec_outputs.argmax(-1).squeeze().tolist()
        decoded_words = []
        for idx in decoded_ids:
            if idx == EOS_token:
                break
            decoded_words.append(output_lang.index2word.get(idx, "<UNK>"))
        
        if show_attention:
            visualize_attention(sentence, decoded_words, attentions[0])
            return attentions[0].cpu().numpy()
        
        # 返回翻译结果作为字符串
        else:
            return ' '.join(decoded_words)

# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from tabulate import tabulate

# 设置字体为 Noto Sans CJK
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # Noto Sans CJK 字体路径
matplotlib.rcParams['font.sans-serif'] = fm.FontProperties(fname=font_path).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

test_cases = [
    ("你好！", "Hello!"),
    ("今天天气如何？", "How is the weather today?"),
    ("我喜欢吃苹果。", "I like to eat apples."),
    ("你能帮我吗？", "Can you help me?"),
    ("现在几点钟？", "What time is it now?"),
    ("这是一个测试。", "This is a test."),
    ("再见！", "Goodbye!"),
    ("我的名字是张三。", "My name is Zhang San."),
    ("你会说英语吗？", "Do you speak English?"),
    ("祝你生日快乐！", "Happy birthday to you!")
]

from tabulate import tabulate

results = []
for src, ref in test_cases:
    translation = enhanced_evaluate(src, show_attention=False)  # 确保不显示注意力
    results.append([src, translation, ref])

# 打印美观的表格
headers = ["Input (Chinese)", "Model Translation", "Reference Translation"]
print(tabulate(results, headers=headers, tablefmt="grid"))


# 获取所有注意力权重矩阵
all_attentions = []
max_len = 0
for src, _ in test_cases:
    attentions = enhanced_evaluate(src, show_attention=True)
    all_attentions.append(attentions)
    max_len = max(max_len, attentions.shape[0])

# 绘制所有注意力权重矩阵
num_cols = 5
num_rows = (len(test_cases) + num_cols - 1) // num_cols  # 计算总行数

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))  # 设置图像大小
axes = axes.flatten()  # 将二维数组展平，方便索引

for i, attentions in enumerate(all_attentions):
    ax = axes[i]
    #attentions_np = attentions[0]  # 获取注意力权重
    if not isinstance(attentions, np.ndarray):  # 检查返回值是否为 NumPy 数组
        ax.axis('off')
        continue
        #raise ValueError(f"Expected attention weights as a NumPy array, but got {attentions}.")
    
    # 填充注意力权重矩阵
    pad_len = max_len - attentions.shape[0]
    if pad_len > 0:
        attentions = np.pad(attentions, ((0, pad_len), (0, pad_len)), 'constant')
    
    ax.matshow(attentions, cmap='viridis')  # 绘制注意力权重矩阵
    ax.axis('off')  # 隐藏坐标轴
    ax.set_title(f"Case {i+1}", fontsize=10)  # 设置标题

# 隐藏多余的子图
for j in range(len(test_cases), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# %%



# %%
