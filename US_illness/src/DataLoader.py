import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class FluDataset:
    def __init__(self, file_path, test_file_path):
        self.file_path = file_path
        self.test_file_path = test_file_path
        self.df_train = None
        self.df_test = None
        self.X = None
        self.y = None
        self.X_test = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self):
        self.df_train = pd.read_excel(self.file_path, engine='openpyxl')
        self.df_test = pd.read_excel(self.test_file_path, engine='openpyxl')
        print(f"训练集加载成功，形状: {self.df_train.shape}")
        print(f"测试集加载成功，形状: {self.df_test.shape}")

    def preprocess_data(self):
        print(f"\n=== 数据预处理 ===")
        self.X, self.y = self._preprocess_data(self.df_train)
        self.X_test, y_waste= self._preprocess_data(self.df_test)

    def _preprocess_data(self, df):
        all_columns = list(df.columns)

        # 1. 分离州编码 (列1-34)
        state_columns = all_columns[1:35]

        # 2. 特征结构
        day1_start = 36
        day1_end = 53
        day2_start = 54
        day2_end = 71
        day3_start = 72
        day3_end = 89

        # 3. 特征名
        feature_names = [
            'cli', 'ili', 'wnohh_cmnty_cli', 'wbelief_masking_effective',
            'wbelief_distancing_effective', 'wcovid_vaccinated_friends',
            'wlarge_event_indoors', 'wothers_masked_public',
            'wothers_distanced_public', 'wshop_indoors',
            'wrestaurant_indoors', 'wworried_catch_covid',
            'hh_cmnty_cli', 'nohh_cmnty_cli', 'wearing_mask_7d',
            'public_transit', 'worried_finances', 'tested_positive'
        ]

        # 4. 构建特征X和目标y
        X_parts = []
        for idx in range(1, len(all_columns) - 1):
            col_name = all_columns[idx]
            X_parts.append(df[col_name].values.reshape(-1, 1))
        X = np.hstack(X_parts).astype(np.float32)
        target_col_name = all_columns[day3_end - 1]
        y = df[target_col_name].values.astype(np.float32)

        # # 6. 构建特征名列表
        # self.feature_names = []
        # self.feature_names.extend([f"state_{col}" for col in state_columns])
        # self.feature_names.extend([f"day1_{feature_names[i]}" for i in range(len(feature_names) - 1)])
        # self.feature_names.extend([f"day2_{feature_names[i]}" for i in range(len(feature_names) - 1)])

        print(f"训练集特征形状: {X.shape}")
        print(f"训练集目标形状: {y.shape}")
        print(f"训练集目标: Day3的 {target_col_name}")

        return X, y


    def create_datasets(self, val_size=0.2):
        # 1. 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=val_size, random_state=42, shuffle=True
        )
        # pd.DataFrame(X_val).to_csv(r'D:\codeC\US_illness\output\val.csv', index=False)

        # 2. 标准化特征（只在训练集上fit）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(self.X_test)

        # 3. 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        print(f"训练集: {X_train_tensor.shape} ({(1 - val_size) * 100:.1f}%)")
        print(f"验证集: {X_val_tensor.shape} ({val_size * 100:.1f}%)")
        print(f"测试集: {X_test_tensor.shape}")
        print(f"总训练样本数: {len(self.X)}")
        print(f"训练样本数: {len(X_train_tensor)}")
        print(f"验证样本数: {len(X_val_tensor)}")
        print(f"测试样本数: {len(X_test_tensor)}")

        return (X_train_tensor, y_train_tensor,
                X_val_tensor, y_val_tensor, X_test_tensor)

if __name__ == "__main__":
    dataset=FluDataset(r"D:\codeC\US_illness\data\train.xlsx",r"D:\codeC\US_illness\data\test.xlsx")
    dataset.load_data()
    dataset.preprocess_data()
    dataset.create_datasets()
