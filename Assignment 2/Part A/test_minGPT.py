#%%
import re, jieba
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN

#%% 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
MAX_LENGTH = 64
BATCH_SIZE = 36  # 增大批大小
TEMPERATURE = 0.7  # 温度参数

#%% 增强版Lang类
class UnifiedLang:
    def __init__(self):
        self.word2index = {SOS_TOKEN: 0, EOS_TOKEN: 1, SEP_TOKEN: 2, "<UNK>": 3}
        self.word2count = {k:1 for k in self.word2index}
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.n_words = 4
        self.special_tokens = {SOS_TOKEN, EOS_TOKEN, SEP_TOKEN}

    def add_sentence(self, sentence, is_chinese=False):
        if is_chinese:
            tokens = jieba.lcut(sentence, cut_all=False) 
        else:
            tokens = sentence.split()
        for token in tokens:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word2index and word not in self.special_tokens:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.word2count[word] = 1
        elif word in self.word2index:
            self.word2count[word] += 1

#%% 数据预处理
def normalize_english(s):
    s = s.strip().lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?.]+", r" ", s)
    return s

def normalize_chinese(s):
    s = s.strip()
    s = re.sub(r"([。！？])", r" \1", s)
    s = re.sub(r"\s+", " ", s)
    return s

def load_dataset(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')[:2]
            if len(parts) == 2:
                eng = normalize_english(parts[0])
                chn = normalize_chinese(parts[1])
                # 过滤掉过长的句子
                # if len(eng.split()) >= 2 and len(chn) >= 2:
                pairs.append((chn, eng))
    return pairs

#%% 数据集类
class TranslationDataset(Dataset):
    def __init__(self, pairs, lang, max_length):
        self.pairs = pairs
        self.lang = lang
        self.max_length = max_length
        self.pad_id = self.lang.word2index["<UNK>"]  # 使用UNK作为填充

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        
        # 构建编码序列
        src_ids = [self.lang.word2index.get(c, 3) for c in src]
        tgt_ids = [self.lang.word2index.get(w, 3) for w in tgt.split()]
        
        # 添加特殊标记
        full_seq = src_ids + [self.lang.word2index[SEP_TOKEN]] + tgt_ids + [self.lang.word2index[EOS_TOKEN]]
        
        # 截断和填充
        seq = full_seq[:self.max_length]
        seq += [self.pad_id]*(self.max_length - len(seq))
        
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

#%% 模型配置
def get_gpt_config(vocab_size):
    config = GPT.get_default_config()
    config.model_type = 'gpt2-medium'
    config.vocab_size = vocab_size
    config.block_size = MAX_LENGTH
    # config.n_layer = 12
    # config.n_head = 16
    config.embd_pdrop = 0.1
    config.resid_pdrop = 0.3  # 增强正则化
    config.attn_pdrop = 0.2
    return config

#%% 训练流程
def train():
    # 初始化统一词汇表
    lang = UnifiedLang()
    pairs = load_dataset('./cmn-eng/cmn.txt')
    
    # 构建词汇表
    for chn, eng in pairs:
        lang.add_sentence(chn, is_chinese=True)
        lang.add_sentence(eng)
    
    # 数据集分割
    train_pairs = pairs
    
    # 创建数据集
    train_dataset = TranslationDataset(train_pairs, lang, MAX_LENGTH)
    
    # 模型配置
    config = get_gpt_config(lang.n_words)
    model = GPT(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 训练配置
    train_config = Trainer.get_default_config()
    train_config.lr_decay = True
    train_config.lr_decay_iters = 200
    train_config.fp16 = True
    train_config.learning_rate = 1e-3  # 降低学习率
    train_config.max_iters = 1000     # 增加训练次数
    train_config.batch_size = BATCH_SIZE
    train_config.grad_norm_clip = 0.5
    train_config.num_workers = 4

        # 初始化进度条
    pbar = tqdm(total=train_config.max_iters, 
                desc="Training Progress",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    # 训练监控
    losses = []
    def loss_callback(trainer):
        loss = trainer.loss.item()
        losses.append(loss)

        # 更新进度条
        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'avg_loss': f"{sum(losses[-100:])/len(losses[-100:]):.4f}" if len(losses) > 0 else 0.0
        })

        if len(losses) % 100 == 0:
            avg_loss = sum(losses[-100:])/100
            # print(f"Iter {len(losses)} | Loss: {avg_loss:.4f}")
    
    # 开始训练
    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', loss_callback)

    try:
        trainer.run()
    finally:
        pbar.close()  # 确保进度条正常关闭
    
    # 保存模型
    torch.save(model.state_dict(), "gpt_translator.pth")
    
    # 绘制损失曲线
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.show()

#%% 增强版生成函数
def translate(model, sentence, lang):
    model.eval()
    token_ids = [lang.word2index.get(c, 3) for c in sentence]
    token_ids.append(lang.word2index[SEP_TOKEN])
    
    generated = []
    for _ in range(MAX_LENGTH):
        inputs = token_ids[-MAX_LENGTH:]  # 滑动窗口
        inputs_tensor = torch.tensor(inputs, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(inputs_tensor)[0][0, -1, :]
        
        # 应用温度采样
        scaled_logits = logits / TEMPERATURE
        probs = torch.softmax(scaled_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        if next_id == lang.word2index[EOS_TOKEN]:
            break
        
        generated.append(next_id)
        token_ids.append(next_id)
    
    # 转换回文本
    tokens = [lang.index2word.get(idx, '<UNK>') for idx in generated]
    return ' '.join(tokens)

#%% 主程序
if __name__ == "__main__":
    # 训练模型
    train()
    
    # 加载模型和词汇表
    pairs = load_dataset('./cmn-eng/cmn.txt')
    lang = UnifiedLang()
    for chn, eng in pairs:
        lang.add_sentence(chn, is_chinese=True)
        lang.add_sentence(eng)
    
    config = get_gpt_config(lang.n_words)
    model = GPT(config).to(device)
    model.load_state_dict(torch.load("gpt_translator.pth"))
    
    # 测试样例
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
    
    print("\nTranslation Results:")
    for src, ref in test_cases:
        translation = translate(model, src, lang)
        print(f"Input: {src}")
        print(f"Output: {translation}")
        print(f"Reference: {ref}\n")