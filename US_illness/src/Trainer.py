import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold


class ModelTrainer:

    def __init__(self, model, device=None):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train(self, X_train, y_train, X_val, y_val,
                     epochs, batch_size,
                     learning_rate, patience):
        """直接训练，不用K折"""
        print("开始训练最终模型...")
        print("=" * 50)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 直接训练最终模型
        history = self._train_single_fold(
            train_loader, val_loader,
            epochs, batch_size, learning_rate, patience
        )

        # 更新结果
        self.train_losses = history['train_losses']
        self.val_losses = history['val_losses']
        self.train_maes = history['train_maes']
        self.val_maes = history['val_maes']
        self.best_val_loss = min(history['val_losses'])

        print(f"\n✓ 模型训练完成，最佳验证MAE: {self.best_val_loss:.4f}")
        return self.model

    def _train_single_fold(self, train_loader, val_loader, epochs, batch_size, learning_rate, patience):
        """训练单个折叠"""
        # 定义损失函数和优化器
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience
        )

        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        patience_counter = 0
        best_fold_val_loss = float('inf')
        best_fold_state = None

        for epoch in range(epochs):
            # 训练
            train_loss, train_mae = self.train_epoch(train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            train_maes.append(train_mae)

            # 验证
            val_loss, val_mae = self.validate(val_loader, criterion)
            val_losses.append(val_loss)
            val_maes.append(val_mae)

            # 调整学习率
            scheduler.step(val_loss)

            # 保存最佳模型（当前折叠内）
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                best_fold_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
                break

        # 加载当前折叠的最佳模型
        if best_fold_state is not None:
            self.model.load_state_dict(best_fold_state)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_maes': train_maes,
            'val_maes': val_maes,
            'best_val_loss': best_fold_val_loss
        }

    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # 计算MAE
            mae = torch.mean(torch.abs(y_pred - y_batch)).item()
            total_loss += loss.item()
            total_mae += mae

        return total_loss / len(train_loader), total_mae / len(train_loader)

    def validate(self, val_loader, criterion):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_mae = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                mae = torch.mean(torch.abs(y_pred - y_batch)).item()
                total_loss += loss.item()
                total_mae += mae

        return total_loss / len(val_loader), total_mae / len(val_loader)
    #
    # def train_with_kfold(self, X_train, y_train, X_val, y_val,
    #                      n_splits=5, epochs=100, batch_size=32,
    #                      learning_rate=0.001, patience=20, random_state=42):
    #     """使用K折交叉验证（仅在训练集上），并使用独立的验证集进行最终评估"""
    #
    #     print(f"开始 {n_splits} 折交叉验证训练...")
    #     print("=" * 60)
    #
    #     # 合并训练集进行K折交叉验证
    #     X_all_train = torch.cat([X_train, X_val], dim=0)
    #     y_all_train = torch.cat([y_train, y_val], dim=0)
    #
    #     # 创建K折交叉验证
    #     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    #     train_dataset = TensorDataset(X_all_train, y_all_train)
    #
    #     fold_val_losses = []
    #     fold_histories = []
    #
    #     for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    #         print(f"\n第 {fold + 1}/{n_splits} 折训练:")
    #         print("-" * 40)
    #
    #         # 创建当前折叠的数据加载器
    #         train_subsampler = Subset(train_dataset, train_idx)
    #         val_subsampler = Subset(train_dataset, val_idx)
    #
    #         train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    #         val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
    #
    #         # 重置模型权重
    #         self.model.apply(self._reset_weights)
    #
    #         # 训练当前折叠
    #         fold_losses = self._train_single_fold(
    #             train_loader, val_loader,
    #             epochs, batch_size, learning_rate, patience
    #         )
    #
    #         fold_val_losses.append(fold_losses['best_val_loss'])
    #         fold_histories.append(fold_losses)
    #
    #         # 更新全局最佳模型
    #         if fold_losses['best_val_loss'] < self.best_val_loss:
    #             self.best_val_loss = fold_losses['best_val_loss']
    #             self.best_model_state = self.model.state_dict().copy()
    #
    #         print(f"  第 {fold + 1} 折完成，最佳验证损失: {fold_losses['best_val_loss']:.4f}")
    #
    #     # 加载最佳模型
    #     if self.best_model_state is not None:
    #         self.model.load_state_dict(self.best_model_state)
    #
    #     # 打印交叉验证结果总结
    #     self._print_kfold_summary(fold_val_losses)
    #
    #     # 在完整的训练集上重新训练最终模型
    #     print("\n使用完整训练集训练最终模型...")
    #     print("-" * 40)
    #
    #     # 创建完整训练集的数据加载器
    #     full_train_dataset = TensorDataset(X_all_train, y_all_train)
    #     full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     # 使用独立的验证集
    #     val_dataset = TensorDataset(X_val, y_val)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #
    #     # 训练最终模型
    #     final_losses = self._train_single_fold(
    #         full_train_loader, val_loader,
    #         epochs, batch_size, learning_rate, patience
    #     )
    #
    #     # 更新最终结果
    #     self.train_losses = final_losses['train_losses']
    #     self.val_losses = final_losses['val_losses']
    #     self.train_maes = final_losses['train_maes']
    #     self.val_maes = final_losses['val_maes']
    #
    #     print(f"\n✓ 最终模型训练完成，验证损失: {self.best_val_loss:.4f}")
    #
    #     return self.model
    #
    # def _train_single_fold(self, train_loader, val_loader, epochs=200, batch_size=32, learning_rate=0.003, patience=20):
    #     """训练单个折叠"""
    #     # 定义损失函数和优化器
    #     criterion = nn.L1Loss()
    #     optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode='min', factor=0.5, patience=10
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
    # def _print_kfold_summary(self, fold_val_losses):
    #     """打印K折交叉验证的总结"""
    #     print("\n" + "=" * 60)
    #     print("K折交叉验证结果总结:")
    #     print("=" * 60)
    #
    #     for i, loss in enumerate(fold_val_losses):
    #         print(f"第 {i + 1} 折验证损失: {loss:.4f}")
    #
    #     print("-" * 60)
    #     print(f"平均验证损失: {np.mean(fold_val_losses):.4f} (±{np.std(fold_val_losses):.4f})")
    #     print(f"最小验证损失: {np.min(fold_val_losses):.4f}")
    #     print(f"最大验证损失: {np.max(fold_val_losses):.4f}")
    #     print("=" * 60)
    #
    # def _reset_weights(self, m):
    #     """重置模型权重（用于每折重新初始化）"""
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
