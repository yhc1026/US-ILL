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

    # def _preprocess_test_data(self, df):
    #     """预处理测试数据"""
    #     all_columns = list(df.columns)
    #
    #     # 1. 分离州编码 (列1-34)
    #     state_columns = all_columns[1:35]
    #
    #     # 2. 测试集只有前两天的完整数据 + 第三天的部分数据
    #     n_features_per_day = 17
    #
    #     # 检查测试集是否有完整的Day3数据
    #     total_cols = len(all_columns)
    #     expected_cols = 1 + 34 + 3 * n_features_per_day  # 86
    #
    #     if total_cols < expected_cols:
    #         # 测试集缺少Day3的tested_positive
    #         print(f"测试集只有{total_cols}列，缺少Day3的tested_positive")
    #
    #         # 计算实际的天数特征
    #         available_cols = total_cols - 35  # 减去id和州编码
    #         # Day1 + Day2 = 34列，剩余的是Day3的特征
    #         day3_cols = available_cols - 34
    #         print(f"Day1: 17列, Day2: 17列, Day3: {day3_cols}列")
    #
    #     # 3. 列索引（测试集可能没有完整的Day3）
    #     day1_start = 35
    #     day1_end = day1_start + n_features_per_day
    #
    #     day2_start = day1_end
    #     day2_end = day2_start + n_features_per_day
    #
    #     # 4. 构建测试特征 X_test（与训练集格式相同）
    #     X_parts = []
    #
    #     # 4.1 州编码
    #     X_parts.append(df[state_columns].values)
    #
    #     # 4.2 Day1特征（前16个，排除tested_positive）
    #     day1_features = []
    #     for i in range(day1_start, day1_end - 1):  # 排除最后一个(tested_positive)
    #         col_name = all_columns[i]
    #         day1_features.append(df[col_name].values.reshape(-1, 1))
    #     day1_matrix = np.hstack(day1_features)
    #     X_parts.append(day1_matrix)
    #
    #     # 4.3 Day2特征（前16个，排除tested_positive）
    #     day2_features = []
    #     for i in range(day2_start, day2_end - 1):  # 排除最后一个(tested_positive)
    #         col_name = all_columns[i]
    #         day2_features.append(df[col_name].values.reshape(-1, 1))
    #     day2_matrix = np.hstack(day2_features)
    #     X_parts.append(day2_matrix)
    #
    #     X_test = np.hstack(X_parts).astype(np.float32)
    #
    #     print(f"测试集特征形状: {X_test.shape}")
    #     print(f"注意: 测试集没有Day3的tested_positive，需要模型预测")
    #
    #     return X_test

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

    def test(self):
        print("=== 检查测试集数据提取 ===")
        print(f"测试集形状: {self.df_test.shape}")
        print(f"测试集列数: {len(self.df_test.columns)}")
        print(f"最后几列名:")
        for i, col in enumerate(self.df_test.columns[-5:], 1):
            print(f"  列{i}: '{col}'")

        # 检查特征矩阵
        print(f"\nX_test形状: {self.X_test.shape}")
        print("X_test前3行前5列:")
        print(self.X_test[:3, :5])
        print("X_test前3行最后5列:")
        print(self.X_test[:3, -5:])
        print("=== 训练集目标变量分析 ===")
        y_train_vals = self.y  # 你的训练集目标

        print(f"训练集y统计:")
        print(f"  最小值: {y_train_vals.min():.4f}")
        print(f"  最大值: {y_train_vals.max():.4f}")
        print(f"  均值: {y_train_vals.mean():.4f}")
        print(f"  中位数: {np.median(y_train_vals):.4f}")
        print(f"  标准差: {y_train_vals.std():.4f}")

        # 查看分布
        plt.figure(figsize=(10, 4))
        plt.hist(y_train_vals, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(y_train_vals.mean(), color='red', linestyle='--', label=f'均值={y_train_vals.mean():.2f}')
        plt.axvline(np.median(y_train_vals), color='green', linestyle='--',
                    label=f'中位数={np.median(y_train_vals):.2f}')
        plt.xlabel('tested_positive')
        plt.ylabel('频次')
        plt.title('训练集目标变量分布')
        plt.legend()
        plt.show()
        wi_col = 'WI'  # 根据你的列名调整
        if wi_col in self.df_train.columns:
            wi_mask = self.df_train[wi_col] == 1
            wi_samples = wi_mask.sum()
            print(f"\n训练集中WI州样本数: {wi_samples}/{len(self.df_train)}")

            if wi_samples > 0:
                wi_y = self.y[wi_mask]
                print(f"WI州的目标变量统计:")
                print(f"  范围: [{wi_y.min():.2f}, {wi_y.max():.2f}]")
                print(f"  均值: {wi_y.mean():.2f}")

    def check_data_leakage_in_detail(dataset):
        """详细检查数据泄露"""

        print("🔍 详细检查数据泄露")
        print("=" * 60)

        # 1. 查看列结构
        print("1. 训练集列结构:")
        train_cols = list(dataset.df_train.columns)
        print(f"   总列数: {len(train_cols)}")
        print(f"   前5列: {train_cols[:5]}")
        print(f"   最后5列: {train_cols[-5:]}")

        # 2. 检查目标列位置
        day3_end = 89  # 根据您的代码
        target_idx = day3_end - 1
        print(f"\n2. 目标列位置检查:")
        print(f"   目标列索引: {target_idx}")
        print(f"   目标列名: '{train_cols[target_idx]}'")

        # 3. 检查特征包含的列范围
        print(f"\n3. 特征包含的列范围:")
        print(f"   特征循环: range(1, {len(train_cols)} - 1)")
        print(f"   实际范围: 列索引 1 到 {len(train_cols) - 2}")
        print(f"   这意味着特征包含了列: {train_cols[1]} 到 '{train_cols[-2]}'")

        # 4. 关键检查：目标列是否在特征中
        if target_idx < len(train_cols) - 1:
            print(f"\n❌❌❌ 严重数据泄露！❌❌❌")
            print(f"   目标列索引: {target_idx}")
            print(f"   特征包含到列索引: {len(train_cols) - 2}")
            print(f"   目标列 '{train_cols[target_idx]}' 被包含在特征中！")
            print(f"   这等于在考试时直接把答案给了模型！")
            return True
        else:
            print(f"\n✅ 目标列不在特征中")
            return False

if __name__ == "__main__":
    dataset=FluDataset(r"D:\codeC\US_illness\data\train.xlsx",r"D:\codeC\US_illness\data\test.xlsx")
    dataset.load_data()
    dataset.preprocess_data()
    dataset.create_datasets()
    dataset.check_data_leakage_in_detail()
#    dataset.test()

