import pandas as pd
import torch



class ModelPredictor:
    """模型预测器"""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X_test):
        """预测测试集"""
        print("\n" + "=" * 60)
        print("5. 预测测试集...")
        print("=" * 60)

        # 确保输入是张量
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.FloatTensor(X_test)

        # 移动到设备
        X_test = X_test.to(self.device)

        # 预测
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy().flatten()

        print(f"✓ 预测完成")
        print(f"  预测样本数: {len(predictions)}")
        print(f"  预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"  预测值均值: {predictions.mean():.4f}")

        return predictions

    def save_predictions(self, predictions, test_df, output_path):
        """保存预测结果"""
        results_df = pd.DataFrame({
            'id': test_df['id'].values,
            'predicted_tested_positive': predictions
        })

        results_df.to_csv(output_path, index=False)
        print(f"\n✓ 预测结果已保存到: {output_path}")