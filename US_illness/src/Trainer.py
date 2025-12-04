import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from Model import FluPredictionModel


class ModelTrainer:

    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 只保留必要的训练历史
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def train_model(self, X_train, y_train, X_val, y_val,
              epochs, batch_size, learning_rate, patience):
        """
        训练模型
        """
        # 创建数据加载器
        train_loader = self._create_data_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, batch_size, shuffle=False)

        # 定义优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()  # 或者用MSELoss()

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = self._train_epoch(train_loader, criterion, optimizer)

            # 验证
            val_loss = self._validate(val_loader, criterion)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
                    break

        # 加载最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

        return self.model

    def _create_data_loader(self, X, y, batch_size, shuffle):
        """创建数据加载器"""
        if isinstance(X, torch.Tensor):
            X_tensor = X
            y_tensor = y
        else:
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    # def __init__(self, model, device=None):
    #     self.model = model
    #     self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     self.model.to(self.device)
    #
    #     self.train_losses = []
    #     self.val_losses = []
    #     self.train_maes = []
    #     self.val_maes = []
    #     self.best_val_loss = float('inf')
    #     self.best_model_state = None
    #
    # def train(self, X_train, y_train, X_val, y_val,
    #                  epochs, batch_size,
    #                  learning_rate, patience):
    #     print("开始训练最终模型...")
    #     print("=" * 50)
    #
    #     # 创建数据加载器
    #     train_dataset = TensorDataset(X_train, y_train)
    #     val_dataset = TensorDataset(X_val, y_val)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #
    #     # 直接训练最终模型
    #     history = self._train_single_fold(
    #         train_loader, val_loader,
    #         epochs, batch_size, learning_rate, patience
    #     )
    #
    #     # 更新结果
    #     self.train_losses = history['train_losses']
    #     self.val_losses = history['val_losses']
    #     self.train_maes = history['train_maes']
    #     self.val_maes = history['val_maes']
    #     self.best_val_loss = min(history['val_losses'])
    #
    #     print(f"\n✓ 模型训练完成，最佳验证MAE: {self.best_val_loss:.4f}")
    #     return self.model
    #
    # def _train_single_fold(self, train_loader, val_loader, epochs, batch_size, learning_rate, patience):
    #     """训练单个折叠"""
    #     # 定义损失函数和优化器
    #     criterion = nn.L1Loss()
    #     optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode='min', factor=0.5, patience=patience
    #     )
    #
    #     train_losses = []
    #     val_losses = []
    #     train_maes = []
    #     val_maes = []
    #     patience_counter = 0
    #     best_fold_val_loss = float('inf')
    #     best_fold_state = None
    #
    #     for epoch in range(epochs):
    #         # 训练
    #         train_loss, train_mae = self.train_epoch(train_loader, criterion, optimizer)
    #         train_losses.append(train_loss)
    #         train_maes.append(train_mae)
    #
    #         # 验证
    #         val_loss, val_mae = self.validate(val_loader, criterion)
    #         val_losses.append(val_loss)
    #         val_maes.append(val_mae)
    #
    #         # 调整学习率
    #         scheduler.step(val_loss)
    #
    #         # 保存最佳模型（当前折叠内）
    #         if val_loss < best_fold_val_loss:
    #             best_fold_val_loss = val_loss
    #             best_fold_state = self.model.state_dict().copy()
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1
    #
    #         # 打印进度
    #         if (epoch + 1) % 10 == 0 or epoch == 0:
    #             print(f"    Epoch {epoch + 1:3d}/{epochs} | "
    #                   f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | "
    #                   f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
    #
    #         # 早停
    #         if patience_counter >= patience:
    #             print(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
    #             break
    #
    #     # 加载当前折叠的最佳模型
    #     if best_fold_state is not None:
    #         self.model.load_state_dict(best_fold_state)
    #
    #     return {
    #         'train_losses': train_losses,
    #         'val_losses': val_losses,
    #         'train_maes': train_maes,
    #         'val_maes': val_maes,
    #         'best_val_loss': best_fold_val_loss
    #     }
    #
    # def train_epoch(self, train_loader, criterion, optimizer):
    #     """训练一个epoch"""
    #     self.model.train()
    #     total_loss = 0
    #     total_mae = 0
    #
    #     for X_batch, y_batch in train_loader:
    #         X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
    #
    #         optimizer.zero_grad()
    #         y_pred = self.model(X_batch)
    #         loss = criterion(y_pred, y_batch)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 计算MAE
    #         mae = torch.mean(torch.abs(y_pred - y_batch)).item()
    #         total_loss += loss.item()
    #         total_mae += mae
    #
    #     return total_loss / len(train_loader), total_mae / len(train_loader)
    #
    # def validate(self, val_loader, criterion):
    #     """验证"""
    #     self.model.eval()
    #     total_loss = 0
    #     total_mae = 0
    #
    #     with torch.no_grad():
    #         for X_batch, y_batch in val_loader:
    #             X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
    #             y_pred = self.model(X_batch)
    #             loss = criterion(y_pred, y_batch)
    #             mae = torch.mean(torch.abs(y_pred - y_batch)).item()
    #             total_loss += loss.item()
    #             total_mae += mae
    #
    #     return total_loss / len(val_loader), total_mae / len(val_loader)