#!/usr/bin/env python3
"""
简化版 AlphaGPT 训练脚本 - 使用合成数据进行快速测试
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# 简单的模型
class SimpleAlphaModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=60, input_size=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_size = input_size

        # 生成合成数据
        self.data = np.random.randn(num_samples, seq_len, input_size)
        self.targets = np.random.randn(num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor([self.targets[idx]])


def train():
    print("="*60)
    print("🚀 AlphaGPT 简化版训练")
    print("="*60)

    # 配置
    device = torch.device('cpu')
    batch_size = 32
    epochs = 3
    learning_rate = 0.001

    print(f"\n配置:")
    print(f"  设备: {device}")
    print(f"  批量大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    print(f"  学习率: {learning_rate}\n")

    # 创建数据
    print("📊 加载数据...")
    dataset = SimpleDataset(num_samples=1000, seq_len=60, input_size=10)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  训练样本: {train_size}")
    print(f"  验证样本: {val_size}\n")

    # 创建模型
    print("🧠 创建模型...")
    model = SimpleAlphaModel(input_size=10, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}\n")

    # 训练
    print("🏋️  开始训练...\n")
    best_loss = float('inf')

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_model_simple.pt')
            print(f"  ✅ 保存最佳模型 (val_loss: {val_loss:.6f})")

        print()

    print("="*60)
    print("✅ 训练完成!")
    print(f"最佳验证损失: {best_loss:.6f}")
    print(f"模型已保存到: best_model_simple.pt")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = train()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
