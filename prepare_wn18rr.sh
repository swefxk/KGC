#!/bin/bash

# 1. 设置工作目录
DATA_DIR="data/wn18rr_custom"
mkdir -p $DATA_DIR
echo "🚀 开始构建 WN18RR 数据集 (国内加速版)..."

# 2. 定义下载函数 (使用加速镜像)
python3 -c "
import os
import urllib.request
import time

# --- 核心修改：使用 GitHub Proxy 加速下载 ---
# 原链接: https://raw.githubusercontent.com/...
# 加速链接: https://mirror.ghproxy.com/https://raw.githubusercontent.com/...
base_url = 'https://mirror.ghproxy.com/https://raw.githubusercontent.com/intfloat/SimKGC/main/data/WN18RR/'

files = ['train.txt', 'valid.txt', 'test.txt', 'entity2text.txt', 'relation2text.txt']
target_dir = '$DATA_DIR'

for file in files:
    url = base_url + file
    save_path = os.path.join(target_dir, file)
    print(f'⬇️  Downloading {file}...')
    
    # 简单的重试机制
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, save_path)
            # 检查文件大小，确保不是空文件
            if os.path.getsize(save_path) > 1000:
                print(f'   ✅ Success: {file}')
                break
            else:
                print('   ⚠️ Downloaded file too small, retrying...')
        except Exception as e:
            print(f'   ❌ Attempt {attempt+1} failed: {e}')
            time.sleep(2)
    else:
        print(f'🔥 Failed to download {file} after 3 attempts.')
        exit(1)
"

# 检查上一步是否成功
if [ $? -ne 0 ]; then
    echo "❌ 下载失败，请检查网络或稍后重试。"
    exit 1
fi

# 3. 核心处理脚本
echo "⚙️  正在处理数据格式..."

python3 -c "
import os

data_dir = '$DATA_DIR'

print('   -> Generating entities.dict and relations.dict...')
entities = set()
relations = set()

try:
    for split in ['train.txt', 'valid.txt', 'test.txt']:
        path = os.path.join(data_dir, split)
        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f'⚠️ Skipping malformed line {line_idx} in {split}')
                    continue
                h, r, t = parts
                entities.add(h)
                entities.add(t)
                relations.add(r)
except FileNotFoundError:
    print('❌ 找不到文件，请确认下载步骤是否成功。')
    exit(1)

sorted_entities = sorted(list(entities))
with open(os.path.join(data_dir, 'entities.dict'), 'w', encoding='utf-8') as f:
    for i, e in enumerate(sorted_entities):
        f.write(f'{i}\t{e}\n')

sorted_relations = sorted(list(relations))
with open(os.path.join(data_dir, 'relations.dict'), 'w', encoding='utf-8') as f:
    for i, r in enumerate(sorted_relations):
        f.write(f'{i}\t{r}\n')

print('   -> Converting text files...')
# 转换 entity2text
if os.path.exists(os.path.join(data_dir, 'entity2text.txt')):
    with open(os.path.join(data_dir, 'entity2text.txt'), 'r', encoding='utf-8') as fin, \
         open(os.path.join(data_dir, 'entity2text_custom.txt'), 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                eid = parts[0]
                text = parts[1]
                name = text.split(': ', 1)[0] if ': ' in text else text
                fout.write(f'{eid}\t{name} [SEP] {text}\n')
    os.replace(os.path.join(data_dir, 'entity2text_custom.txt'), os.path.join(data_dir, 'entity2text.txt'))

# 转换 relation2text
if os.path.exists(os.path.join(data_dir, 'relation2text.txt')):
    with open(os.path.join(data_dir, 'relation2text.txt'), 'r', encoding='utf-8') as fin, \
         open(os.path.join(data_dir, 'relation2text_custom.txt'), 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                fout.write(f'{parts[0]}\t{parts[0]} [SEP] {parts[1]}\n')
    os.replace(os.path.join(data_dir, 'relation2text_custom.txt'), os.path.join(data_dir, 'relation2text.txt'))

print('✅ Data processing complete!')
"