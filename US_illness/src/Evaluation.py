import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from Trainer import ModelTrainer
from Model import FluPredictionModel

class Validation:
    def __init__(self, X_train, y_train, X_val, y_val,epochs, batch_size, learning_rate, patience, input_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.input_size = input_size

    def five_fold_cross_validation(self, n_splits=5, random_state=None):
        X_full = np.vstack([self.X_train, self.X_val])
        y_full = np.concatenate([self.y_train, self.y_val])

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        mae_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
            print(f"fold: {fold+1}")
            model = FluPredictionModel(self.input_size)
            # 划分当前折的训练集和验证集
            X_train_fold = X_full[train_idx]
            y_train_fold = y_full[train_idx]
            X_val_fold = X_full[val_idx]
            y_val_fold = y_full[val_idx]

            # 训练模型
            trainer = ModelTrainer(model)
            trained_model=trainer.train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold,self.epochs, self.batch_size, self.learning_rate, self.patience)

            X_val_tensor = torch.FloatTensor(X_val_fold).to(trainer.device)

            # 预测并计算MAE
            with torch.no_grad():
                trained_model.eval()
                y_pred_tensor = trained_model(X_val_tensor)
                y_pred = y_pred_tensor.cpu().numpy()
            mae = mean_absolute_error(y_val_fold, y_pred)
            mae_scores.append(mae)

        # 计算平均MAE
        mean_mae = np.mean(mae_scores)

        return mean_mae
