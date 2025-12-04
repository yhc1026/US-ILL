import torch

from DataLoader import FluDataset
from Model import FluPredictionModel
from Prediction import ModelPredictor
from Trainer import ModelTrainer
from Evaluation import Validation
from datetime import datetime
import pandas as pd
import numpy as np

epochs = 200
batch_size = 32
learning_rate = 0.0006
patience = 25

def predict():
    train_path = r'D:\codeC\US-ILL\US_illness\data\train_1.xlsx'
    test_path = r'D:\codeC\US-ILL\US_illness\data\test_1.xlsx'

    n_ensemble = 10

    # 1.处理数据
    processor = FluDataset(train_path, test_path)
    processor.load_data()
    processor.preprocess_data()
    X_train, y_train, X_val, y_val, X_test = processor.create_datasets()

    print(f"训练 {n_ensemble} 个模型取平均")
    print("=" * 60)

    all_predictions = []  # 存储所有模型的预测

    for i in range(n_ensemble):
        print(f"\n训练第 {i + 1}/{n_ensemble} 个模型...")

        # 2. 每次循环重新构建模型（关键！）
        input_size = X_train.shape[1]
        model = FluPredictionModel(input_size)

        # 3. 创建新的训练器
        trainer = ModelTrainer(model)

        # 4. 训练模型
        trained_model = trainer.train_model(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, patience=patience
        )

        # 5. 预测
        predictor = ModelPredictor(trained_model, trainer.device)
        predictions = predictor.predict(X_test)

        # 6. 记录结果
        all_predictions.append(predictions)

    # 7. 计算平均预测
    # 将预测列表转为numpy数组
    all_predictions_array = np.array(all_predictions)
    ensemble_predictions = np.mean(all_predictions_array, axis=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'D:\\codeC\\US-ILL\\US_illness\\output\\flu_predictions_ensemble_{timestamp}.csv'
    results_df = pd.DataFrame({
        'id': processor.df_test['id'].values if 'id' in processor.df_test.columns else range(len(ensemble_predictions)),
        'predicted_tested_positive': ensemble_predictions
    })
    results_df['learning_rate'] = learning_rate
    results_df['epochs'] = epochs
    results_df['patience'] = patience
    results_df['batch_size'] = batch_size
    results_df['n_ensemble_models'] = n_ensemble
    results_df['ensemble_method'] = 'mean_average'
    results_df['generation_time'] = timestamp
    results_df.to_csv(output_path, index=False)
    test=[18.4907873,16.3292532,16.5229315,15.578501,14.1719204,13.9259455,14.36337,15.5106489,14.5280436,13.6732144]
    abs_errors = torch.abs(torch.tensor(test) - torch.tensor(ensemble_predictions))
    mae = torch.mean(abs_errors)
    mae_item=mae.item()
    print(f"预测范围: {ensemble_predictions.min():.2f} ~ {ensemble_predictions.max():.2f}, MAE: {mae_item:.2f}")


def validate():
    MAEs=[]
    for i in range(5):
        print(f"MAE计算第{i+1}轮")
        train_path = r'D:\codeC\US-ILL\US_illness\data\train.xlsx'
        test_path = r'D:\codeC\US-ILL\US_illness\data\test.xlsx'

        processor = FluDataset(train_path, test_path)
        processor.load_data()
        processor.preprocess_data()
        X_train, y_train, X_val, y_val, X_test = processor.create_datasets()
        input_size = X_train.shape[1]
        print("\n")
        print("\n")
        validate=Validation(X_train,y_train,X_val,y_val,epochs, batch_size, learning_rate, patience, input_size)
        MAE=validate.five_fold_cross_validation()
        MAEs.append(MAE)

    print(f"MAEs: {MAEs}\n")
    MAE_mean=np.mean(MAEs)
    print(f"epoche={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, patience={patience},MAE_mean: {MAE_mean}")


if __name__ == "__main__":
    predict()
    #validate()
