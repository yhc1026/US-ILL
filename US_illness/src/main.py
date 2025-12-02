from DataLoader import FluDataset
from Model import FluPredictionModel
from Prediction import ModelPredictor
from Trainer import ModelTrainer
from Evaluation import ModelEvaluator
from datetime import datetime
import pandas as pd
import numpy as np


def run():
    train_path = r'D:\codeC\US_illness\data\train.xlsx'
    test_path = r'D:\codeC\US_illness\data\test.xlsx'

    epochs = 200
    batch_size = 32
    learning_rate = 0.005
    patience = 25
    n_ensemble = 10

    # 1.处理数据
    processor = FluDataset(train_path, test_path)
    processor.load_data()
    processor.preprocess_data()
    X_train, y_train, X_val, y_val, X_test = processor.create_datasets()

    print(f"开始集成训练，将训练 {n_ensemble} 个模型取平均...")
    print("=" * 60)

    all_predictions = []  # 存储所有模型的预测
    model_performances = []  # 记录每个模型的性能

    for i in range(n_ensemble):
        print(f"\n训练第 {i + 1}/{n_ensemble} 个模型...")

        # 2. 每次循环重新构建模型（关键！）
        input_size = X_train.shape[1]
        model = FluPredictionModel(input_size)

        # 3. 创建新的训练器
        trainer = ModelTrainer(model)

        # 4. 训练模型
        trained_model = trainer.train(
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
        model_performances.append({
            'model_id': i + 1,
            'best_val_mae': trainer.best_val_loss,
            'final_train_mae': trainer.train_losses[-1] if trainer.train_losses else None,
            'final_val_mae': trainer.val_losses[-1] if trainer.val_losses else None
        })

        print(f"  模型 {i + 1} 完成，最佳验证MAE: {trainer.best_val_loss:.4f}")

    # 7. 计算平均预测
    print(f"\n{'=' * 60}")
    print("计算集成平均预测...")

    # 将预测列表转为numpy数组
    all_predictions_array = np.array(all_predictions)  # 形状: (10, 测试样本数)

    # 计算平均值（集成预测）
    ensemble_predictions = np.mean(all_predictions_array, axis=0)

    # 8. 分析集成效果
    print(f"集成模型分析:")
    print(f"训练了 {len(all_predictions)} 个模型")

    # 计算单个模型的性能统计
    val_maes = [perf['best_val_mae'] for perf in model_performances]
    print(f"单个模型验证MAE范围: {min(val_maes):.4f} ~ {max(val_maes):.4f}")
    print(f"单个模型平均验证MAE: {np.mean(val_maes):.4f} (±{np.std(val_maes):.4f})")

    # 9. 保存集成结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'D:\\codeC\\US_illness\\output\\flu_predictions_ensemble_{timestamp}.csv'

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'id': processor.df_test['id'].values if 'id' in processor.df_test.columns else range(len(ensemble_predictions)),
        'predicted_tested_positive': ensemble_predictions
    })

    # 添加模型和训练信息
    results_df['learning_rate'] = learning_rate
    results_df['epochs'] = epochs
    results_df['patience'] = patience
    results_df['batch_size'] = batch_size
    results_df['n_ensemble_models'] = n_ensemble
    results_df['ensemble_method'] = 'mean_average'
    results_df['generation_time'] = timestamp

    # 保存
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ 集成预测结果已保存到: {output_path}")
    print(f"预测样本数: {len(ensemble_predictions)}")
    print(f"预测范围: {ensemble_predictions.min():.2f} ~ {ensemble_predictions.max():.2f}")


# def run():
#     train_path = r'D:\codeC\US_illness\data\train.xlsx'
#     test_path = r'D:\codeC\US_illness\data\test.xlsx'
#
#     epochs = 200
#     batch_size = 32
#     learning_rate = 0.003
#     patience = 20
#
#     # 1.处理数据
#     processor=FluDataset(train_path,test_path)
#     processor.load_data()
#     processor.preprocess_data()
#     X_train, y_train,X_val, y_val, X_test=processor.create_datasets()
#
#     # 2. 构建模型
#     input_size = X_train.shape[1]
#     model = FluPredictionModel(input_size)
#
#     # 3. 训练模型
#     trainer = ModelTrainer(model)
#     trained_model = trainer.train(
#         X_train=X_train, y_train=y_train,
#         X_val=X_val, y_val=y_val,
#         epochs=epochs, batch_size=batch_size,
#         learning_rate=learning_rate, patience=patience
#     )
#
#     # 4. 评估模型性能
#     #evaluator=ModelEvaluator(trained_model)
#     #report = evaluator.generate_report(
#         #     X_train=X_train,  # 训练数据
#         #     y_train=y_train,  # 训练标签
#         #     X_val=X_val,      # 验证数据
#         #     y_val=y_val,      # 验证标签
#         # )
#
#     # 5. 预测测试集并保存
#
#     predictor = ModelPredictor(trained_model, trainer.device)
#     predictions = predictor.predict(X_test)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_path = f'D:\\codeC\\US_illness\\output\\flu_predictions_{timestamp}.csv'
#     predictor.save_predictions(predictions, processor.df_test, output_path)
#     df = pd.read_csv(output_path)
#     df['learning_rate']=learning_rate
#     df['epochs']=epochs
#     df['patience']=patience
#     df['batch_size']=batch_size
#     df.to_csv(output_path, index=False)



if __name__ == "__main__":
    run()
    print("\n运行成功！")