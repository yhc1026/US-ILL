import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class ModelEvaluator:
    """
    完整的模型评估工具类
    适用于您的回归预测问题
    """

    def __init__(self, model, trainer=None, device=None):
        """
        初始化评估器

        Parameters:
        -----------
        model : torch.nn.Module
            训练好的PyTorch模型
        trainer : ModelTrainer, optional
            您的训练器对象，用于获取训练历史
        device : torch.device, optional
            设备（CPU/GPU）
        """
        self.model = model
        self.trainer = trainer
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.metrics = {}
        self.predictions = {}

    def predict(self, X):
        """生成预测"""
        if not torch.is_tensor(X):
            X = torch.FloatTensor(X)

        X = X.to(self.device)
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(X)

        return y_pred.cpu().numpy()

    def compute_metrics(self, y_true, y_pred, prefix='val'):
        """计算所有评估指标"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # 基本回归指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # 百分比误差
        absolute_percentage_error = np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10)) * 100
        mape = np.mean(absolute_percentage_error)

        # 对称MAPE
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

        # 最大最小误差
        max_error = np.max(np.abs(y_true - y_pred))
        min_error = np.min(np.abs(y_true - y_pred))

        # 误差统计
        errors = y_true - y_pred
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        median_error = np.median(np.abs(errors))

        # 精度比例
        thresholds = [1, 5, 10, 15, 20]
        accuracy_rates = {}
        for threshold in thresholds:
            accuracy_rates[f'within_{threshold}pct'] = np.mean(absolute_percentage_error <= threshold) * 100
            accuracy_rates[f'within_{threshold}_abs'] = np.mean(np.abs(errors) <= threshold) * 100

        metrics = {
            f'{prefix}_mae': mae,
            f'{prefix}_mse': mse,
            f'{prefix}_rmse': rmse,
            f'{prefix}_r2': r2,
            f'{prefix}_mape': mape,
            f'{prefix}_smape': smape,
            f'{prefix}_max_error': max_error,
            f'{prefix}_min_error': min_error,
            f'{prefix}_mean_error': mean_error,
            f'{prefix}_std_error': std_error,
            f'{prefix}_median_abs_error': median_error,
            f'{prefix}_errors': errors,
            f'{prefix}_percentage_errors': absolute_percentage_error,
            f'{prefix}_y_true': y_true,
            f'{prefix}_y_pred': y_pred,
        }

        metrics.update(accuracy_rates)
        self.metrics.update(metrics)

        return metrics

    def compare_with_baselines(self, y_true, X_data=None):
        """与基准模型比较"""
        y_true = np.array(y_true).flatten()

        baselines = {}

        # 基准1：均值预测
        baseline_mean = np.mean(y_true)
        baselines['mean'] = {
            'mae': mean_absolute_error(y_true, np.full_like(y_true, baseline_mean)),
            'rmse': np.sqrt(mean_squared_error(y_true, np.full_like(y_true, baseline_mean))),
            'predictions': np.full_like(y_true, baseline_mean)
        }

        # 基准2：中位数预测
        baseline_median = np.median(y_true)
        baselines['median'] = {
            'mae': mean_absolute_error(y_true, np.full_like(y_true, baseline_median)),
            'rmse': np.sqrt(mean_squared_error(y_true, np.full_like(y_true, baseline_median))),
            'predictions': np.full_like(y_true, baseline_median)
        }

        # 基准3：最后一个已知值（适用于时间序列）
        if X_data is not None and hasattr(X_data, 'shape'):
            # 假设最后一个特征是y_b（时间点b的值）
            try:
                if torch.is_tensor(X_data):
                    y_b = X_data[:, -1].cpu().numpy()  # 假设最后一列是y_b
                else:
                    y_b = X_data[:, -1]  # 假设最后一列是y_b

                baselines['last_value'] = {
                    'mae': mean_absolute_error(y_true, y_b),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_b)),
                    'predictions': y_b
                }
            except:
                pass

        # 基准4：线性外推（如果知道y_a和y_b）
        if X_data is not None and X_data.shape[1] >= 2:
            try:
                if torch.is_tensor(X_data):
                    y_a = X_data[:, -2].cpu().numpy()
                    y_b = X_data[:, -1].cpu().numpy()
                else:
                    y_a = X_data[:, -2]
                    y_b = X_data[:, -1]

                # 线性外推：y_c = y_b + (y_b - y_a)
                linear_extrapolation = 2 * y_b - y_a
                baselines['linear_extrapolation'] = {
                    'mae': mean_absolute_error(y_true, linear_extrapolation),
                    'rmse': np.sqrt(mean_squared_error(y_true, linear_extrapolation)),
                    'predictions': linear_extrapolation
                }
            except:
                pass

        return baselines

    def plot_training_history(self):
        """绘制训练历史曲线"""
        if self.trainer is None:
            print("警告：未提供trainer对象，无法绘制训练历史")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 损失曲线
        axes[0, 0].plot(self.trainer.train_losses, label='训练损失', linewidth=2, alpha=0.8)
        axes[0, 0].plot(self.trainer.val_losses, label='验证损失', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[0, 0].set_ylabel('损失 (MSE)', fontsize=12)
        axes[0, 0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        # 标记最佳epoch
        if hasattr(self.trainer, 'best_val_loss'):
            best_idx = np.argmin(self.trainer.val_losses)
            axes[0, 0].axvline(x=best_idx, color='red', linestyle='--', alpha=0.7,
                               label=f'最佳epoch: {best_idx + 1}')
            axes[0, 0].scatter(best_idx, self.trainer.val_losses[best_idx],
                               color='red', s=100, zorder=5)

        # 2. MAE曲线
        axes[0, 1].plot(self.trainer.train_maes, label='训练MAE', linewidth=2, alpha=0.8)
        axes[0, 1].plot(self.trainer.val_maes, label='验证MAE', linewidth=2, alpha=0.8)
        axes[0, 1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[0, 1].set_ylabel('MAE', fontsize=12)
        axes[0, 1].set_title('训练和验证MAE曲线', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 损失对数图
        axes[1, 0].semilogy(self.trainer.train_losses, label='训练损失', linewidth=2, alpha=0.8)
        axes[1, 0].semilogy(self.trainer.val_losses, label='验证损失', linewidth=2, alpha=0.8)
        axes[1, 0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[1, 0].set_ylabel('损失 (对数尺度)', fontsize=12)
        axes[1, 0].set_title('损失曲线（对数尺度）', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 过拟合分析
        if len(self.trainer.train_losses) > 0 and len(self.trainer.val_losses) > 0:
            overfitting_ratio = np.array(self.trainer.val_losses) / np.array(self.trainer.train_losses)
            axes[1, 1].plot(overfitting_ratio, label='验证损失/训练损失', linewidth=2,
                            color='purple', alpha=0.8)
            axes[1, 1].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='理想线')
            axes[1, 1].axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='过拟合警戒线')
            axes[1, 1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[1, 1].set_ylabel('损失比例', fontsize=12)
        axes[1, 1].set_title('过拟合分析', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('模型训练历史分析', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_prediction_analysis(self, y_true, y_pred):
        """绘制预测分析图"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        fig = plt.figure(figsize=(20, 16))

        # 1. 预测 vs 真实值散点图
        ax1 = plt.subplot(3, 3, 1)
        scatter = ax1.scatter(y_true, y_pred, alpha=0.6, s=30, c=np.abs(y_true - y_pred),
                              cmap='viridis', edgecolors='black', linewidth=0.5)
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='完美预测线')
        ax1.set_xlabel('真实值', fontsize=12)
        ax1.set_ylabel('预测值', fontsize=12)
        ax1.set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('绝对误差', fontsize=11)

        # 2. 残差图
        ax2 = plt.subplot(3, 3, 2)
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, s=30, c=np.abs(residuals),
                    cmap='coolwarm', edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=3)
        ax2.set_xlabel('预测值', fontsize=12)
        ax2.set_ylabel('残差 (真实值 - 预测值)', fontsize=12)
        ax2.set_title('残差图', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. 残差分布
        ax3 = plt.subplot(3, 3, 3)
        n, bins, patches = ax3.hist(residuals, bins=40, edgecolor='black',
                                    alpha=0.7, color='skyblue')
        ax3.axvline(x=0, color='r', linestyle='--', lw=3)
        ax3.set_xlabel('残差', fontsize=12)
        ax3.set_ylabel('频数', fontsize=12)
        ax3.set_title('残差分布直方图', fontsize=14, fontweight='bold')

        # 添加正态分布曲线
        mu, std = stats.norm.fit(residuals)
        xmin, xmax = ax3.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax3.plot(x, p * len(residuals) * (bins[1] - bins[0]), 'r-', lw=2,
                 label=f'正态分布拟合\nμ={mu:.2f}, σ={std:.2f}')
        ax3.legend(fontsize=10)

        # 4. 百分比误差分布
        ax4 = plt.subplot(3, 3, 4)
        percentage_errors = np.abs(residuals / (np.abs(y_true) + 1e-10)) * 100
        n, bins, patches = ax4.hist(percentage_errors, bins=40, edgecolor='black',
                                    alpha=0.7, color='lightcoral')

        # 添加百分比线
        for threshold in [10, 20, 30]:
            color = 'green' if threshold == 10 else 'orange' if threshold == 20 else 'red'
            ax4.axvline(x=threshold, color=color, linestyle='--', lw=2,
                        label=f'{threshold}%误差线')

        ax4.set_xlabel('绝对百分比误差 (%)', fontsize=12)
        ax4.set_ylabel('频数', fontsize=12)
        ax4.set_title('百分比误差分布', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)

        # 5. 累积误差分布
        ax5 = plt.subplot(3, 3, 5)
        sorted_abs_errors = np.sort(np.abs(residuals))
        cumulative_prop = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)

        ax5.plot(sorted_abs_errors, cumulative_prop * 100, lw=3, color='darkblue')

        # 标记关键点
        for pct in [50, 80, 90, 95]:
            idx = int(pct / 100 * len(sorted_abs_errors)) - 1
            if idx >= 0:
                ax5.scatter(sorted_abs_errors[idx], pct, color='red', s=100, zorder=5)
                ax5.annotate(f'{pct}%: {sorted_abs_errors[idx]:.2f}',
                             xy=(sorted_abs_errors[idx], pct),
                             xytext=(10, 10), textcoords='offset points',
                             fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor="yellow", alpha=0.7))

        ax5.set_xlabel('绝对误差', fontsize=12)
        ax5.set_ylabel('累积百分比 (%)', fontsize=12)
        ax5.set_title('累积误差分布', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. QQ图（正态性检验）
        ax6 = plt.subplot(3, 3, 6)
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
        ax6.plot(osm, osr, 'o', alpha=0.6, markersize=6, markeredgecolor='black',
                 markeredgewidth=0.5)
        ax6.plot(osm, slope * osm + intercept, 'r-', lw=3,
                 label=f'拟合线 (R={r:.3f})')
        ax6.set_xlabel('理论分位数', fontsize=12)
        ax6.set_ylabel('样本分位数', fontsize=12)
        ax6.set_title('QQ图 - 残差正态性检验', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)

        # 7. 误差随时间/顺序变化
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(np.abs(residuals), alpha=0.7, lw=2, color='darkgreen')
        ax7.axhline(y=np.mean(np.abs(residuals)), color='red',
                    linestyle='--', lw=2, label=f'平均绝对误差: {np.mean(np.abs(residuals)):.2f}')
        ax7.fill_between(range(len(residuals)), 0, np.abs(residuals),
                         alpha=0.3, color='lightgreen')
        ax7.set_xlabel('样本序号', fontsize=12)
        ax7.set_ylabel('绝对误差', fontsize=12)
        ax7.set_title('误差随时间/顺序变化', fontsize=14, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)

        # 8. 误差箱线图
        ax8 = plt.subplot(3, 3, 8)
        bp = ax8.boxplot([residuals, np.abs(residuals), percentage_errors],
                         labels=['残差', '绝对误差', '百分比误差(%)'],
                         patch_artist=True,
                         medianprops=dict(color='black', linewidth=2),
                         boxprops=dict(facecolor='lightblue', alpha=0.7))

        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax8.set_ylabel('值', fontsize=12)
        ax8.set_title('误差统计箱线图', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. 预测值分布对比
        ax9 = plt.subplot(3, 3, 9)
        bins = np.linspace(min(min(y_true), min(y_pred)),
                           max(max(y_true), max(y_pred)), 30)
        ax9.hist(y_true, bins=bins, alpha=0.5, label='真实值', color='blue',
                 edgecolor='black', density=True)
        ax9.hist(y_pred, bins=bins, alpha=0.5, label='预测值', color='red',
                 edgecolor='black', density=True)
        ax9.set_xlabel('值', fontsize=12)
        ax9.set_ylabel('密度', fontsize=12)
        ax9.set_title('真实值与预测值分布对比', fontsize=14, fontweight='bold')
        ax9.legend(fontsize=11)

        plt.suptitle('模型预测分析报告', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

        return residuals, percentage_errors

    def plot_baseline_comparison(self, y_true, y_pred, baselines):
        """绘制与基准模型的比较"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        models = ['您的模型']
        mae_scores = [mean_absolute_error(y_true, y_pred)]
        rmse_scores = [np.sqrt(mean_squared_error(y_true, y_pred))]

        for name, baseline in baselines.items():
            models.append(name)
            mae_scores.append(baseline['mae'])
            rmse_scores.append(baseline['rmse'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # MAE比较
        bars1 = axes[0].bar(models, mae_scores, color=['blue'] + ['gray'] * len(baselines),
                            alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('模型', fontsize=12)
        axes[0].set_ylabel('MAE', fontsize=12)
        axes[0].set_title('MAE比较 (越低越好)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值
        for i, (bar, score) in enumerate(zip(bars1, mae_scores)):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        # 计算提升百分比
        improvement = []
        for i, score in enumerate(mae_scores[1:], 1):
            improv_pct = (mae_scores[0] - score) / score * 100
            improvement.append(improv_pct)
            axes[0].text(i, mae_scores[i] * 1.05,
                         f'{improv_pct:+.1f}%' if i > 0 else '',
                         ha='center', va='bottom', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # RMSE比较
        bars2 = axes[1].bar(models, rmse_scores, color=['blue'] + ['gray'] * len(baselines),
                            alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('模型', fontsize=12)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE比较 (越低越好)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值
        for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        plt.suptitle('模型性能与基准对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return models, mae_scores, rmse_scores

    def generate_report(self, X_train=None, y_train=None, X_val=None, y_val=None,
                        X_test=None, y_test=None, feature_names=None):
        """生成完整的评估报告"""
        print("=" * 80)
        print("🤖 机器学习模型评估报告")
        print("=" * 80)

        # 存储所有结果
        results = {}

        # 1. 训练历史分析
        if self.trainer:
            print("\n📊 1. 训练历史分析")
            print("-" * 40)

            train_losses = np.array(self.trainer.train_losses)
            val_losses = np.array(self.trainer.val_losses)

            best_epoch = np.argmin(val_losses) + 1
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]

            print(f"   总训练轮次: {len(train_losses)}")
            print(f"   最佳验证轮次: {best_epoch}")
            print(f"   最终训练损失: {final_train_loss:.6f}")
            print(f"   最终验证损失: {final_val_loss:.6f}")
            print(f"   最佳验证损失: {np.min(val_losses):.6f}")

            # 收敛分析
            if len(val_losses) >= 10:
                last_10_std = np.std(val_losses[-10:])
                print(f"   最后10轮损失标准差: {last_10_std:.6f}")
                if last_10_std < 0.001 * np.mean(val_losses):
                    print("   ✅ 模型已收敛")
                else:
                    print("   ⚠️  模型可能未完全收敛")

            # 过拟合分析
            overfitting_ratio = final_val_loss / final_train_loss
            print(f"   验证/训练损失比: {overfitting_ratio:.3f}")
            if overfitting_ratio > 1.5:
                print("   ⚠️  警告：可能过拟合")
            elif overfitting_ratio < 1.1:
                print("   ✅ 良好：欠拟合风险低")
            else:
                print("   ⚠️  注意：有一定过拟合迹象")

            results['training_history'] = {
                'best_epoch': best_epoch,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'best_val_loss': np.min(val_losses),
                'overfitting_ratio': overfitting_ratio
            }

        # 2. 验证集评估
        if X_val is not None and y_val is not None:
            print("\n📈 2. 验证集性能评估")
            print("-" * 40)

            # 生成预测
            y_val_pred = self.predict(X_val)
            y_val_true = np.array(y_val).flatten()

            # 计算指标
            val_metrics = self.compute_metrics(y_val_true, y_val_pred, 'val')

            print(f"   MAE: {val_metrics['val_mae']:.4f}")
            print(f"   RMSE: {val_metrics['val_rmse']:.4f}")
            print(f"   R²: {val_metrics['val_r2']:.4f}")
            print(f"   MAPE: {val_metrics['val_mape']:.2f}%")
            print(f"   SMAPE: {val_metrics['val_smape']:.2f}%")
            print(f"   最大误差: {val_metrics['val_max_error']:.4f}")
            print(f"   误差标准差: {val_metrics['val_std_error']:.4f}")

            # 精度统计
            print(f"\n   📊 预测精度:")
            for threshold in [5, 10, 15, 20]:
                key = f'val_within_{threshold}pct'
                if key in val_metrics:
                    print(f"     误差在±{threshold}%以内: {val_metrics[key]:.1f}%")

            results['validation_metrics'] = val_metrics

            # 绘制预测分析图
            print("\n   📉 绘制预测分析图...")
            residuals, percentage_errors = self.plot_prediction_analysis(y_val_true, y_val_pred)

            # 与基准比较
            print("\n   📊 与基准模型比较...")
            baselines = self.compare_with_baselines(y_val_true, X_val)
            self.plot_baseline_comparison(y_val_true, y_val_pred, baselines)

            # 打印基准比较
            print("\n     基准模型性能:")
            for name, baseline in baselines.items():
                print(f"     {name}: MAE={baseline['mae']:.4f}, "
                      f"RMSE={baseline['rmse']:.4f}")

                # 计算提升
                improv_mae = (baseline['mae'] - val_metrics['val_mae']) / baseline['mae'] * 100
                improv_rmse = (baseline['rmse'] - val_metrics['val_rmse']) / baseline['rmse'] * 100
                print(f"       相对提升: MAE={improv_mae:+.1f}%, RMSE={improv_rmse:+.1f}%")

        # 3. 测试集评估（如果有）
        if X_test is not None and y_test is not None:
            print("\n🧪 3. 测试集性能评估")
            print("-" * 40)

            y_test_pred = self.predict(X_test)
            y_test_true = np.array(y_test).flatten()

            test_metrics = self.compute_metrics(y_test_true, y_test_pred, 'test')

            print(f"   MAE: {test_metrics['test_mae']:.4f}")
            print(f"   RMSE: {test_metrics['test_rmse']:.4f}")
            print(f"   R²: {test_metrics['test_r2']:.4f}")
            print(f"   MAPE: {test_metrics['test_mape']:.2f}%")

            # 与验证集比较
            if 'validation_metrics' in results:
                print(f"\n   🔄 与验证集比较:")
                mae_diff = test_metrics['test_mae'] - results['validation_metrics']['val_mae']
                r2_diff = test_metrics['test_r2'] - results['validation_metrics']['val_r2']

                print(f"     MAE变化: {mae_diff:+.4f}")
                print(f"     R²变化: {r2_diff:+.4f}")

                if abs(mae_diff) > 0.1 * results['validation_metrics']['val_mae']:
                    print("     ⚠️  注意：测试集与验证集性能差异较大")
                else:
                    print("     ✅ 良好：测试集与验证集性能一致")

            results['test_metrics'] = test_metrics

        # 4. 训练集评估（检查过拟合）
        if X_train is not None and y_train is not None:
            print("\n🎓 4. 训练集性能（检查过拟合）")
            print("-" * 40)

            y_train_pred = self.predict(X_train)
            y_train_true = np.array(y_train).flatten()

            train_metrics = self.compute_metrics(y_train_true, y_train_pred, 'train')

            print(f"   MAE: {train_metrics['train_mae']:.4f}")
            print(f"   R²: {train_metrics['train_r2']:.4f}")

            if 'validation_metrics' in results:
                gap_mae = train_metrics['train_mae'] - results['validation_metrics']['val_mae']
                gap_r2 = results['validation_metrics']['val_r2'] - train_metrics['train_r2']

                print(f"\n   🔍 训练集-验证集差距:")
                print(f"     MAE差距: {gap_mae:+.4f} "
                      f"(负值表示过拟合)")
                print(f"     R²差距: {gap_r2:+.4f} "
                      f"(正值表示过拟合)")

                if gap_mae < -0.1 * results['validation_metrics']['val_mae']:
                    print("     ⚠️  可能过拟合：训练集性能明显优于验证集")
                elif gap_mae > 0.1 * results['validation_metrics']['val_mae']:
                    print("     ⚠️  可能欠拟合：训练集性能不如验证集")
                else:
                    print("     ✅ 良好：训练集和验证集性能平衡")

            results['train_metrics'] = train_metrics

        # 5. 性能总结
        print("\n" + "=" * 80)
        print("📋 5. 性能总结与建议")
        print("=" * 80)

        # 收集关键指标
        key_metrics = {}
        if 'validation_metrics' in results:
            key_metrics.update({
                'val_mae': results['validation_metrics']['val_mae'],
                'val_r2': results['validation_metrics']['val_r2'],
                'val_mape': results['validation_metrics']['val_mape']
            })

        if 'test_metrics' in results:
            key_metrics.update({
                'test_mae': results['test_metrics']['test_mae'],
                'test_r2': results['test_metrics']['test_r2']
            })

        # 判断模型质量
        recommendations = []

        if 'val_r2' in key_metrics:
            r2 = key_metrics['val_r2']
            if r2 >= 0.9:
                recommendations.append("✅ 模型解释力极强 (R² ≥ 0.9)")
            elif r2 >= 0.7:
                recommendations.append("✅ 模型解释力良好 (0.7 ≤ R² < 0.9)")
            elif r2 >= 0.5:
                recommendations.append("⚠️  模型解释力一般 (0.5 ≤ R² < 0.7)，可考虑优化特征")
            else:
                recommendations.append("❌ 模型解释力不足 (R² < 0.5)，需要重新设计模型")

        if 'val_mape' in key_metrics:
            mape = key_metrics['val_mape']
            if mape <= 10:
                recommendations.append("✅ 预测精度极高 (MAPE ≤ 10%)")
            elif mape <= 20:
                recommendations.append("✅ 预测精度良好 (10% < MAPE ≤ 20%)")
            elif mape <= 30:
                recommendations.append("⚠️  预测精度一般 (20% < MAPE ≤ 30%)")
            else:
                recommendations.append("❌ 预测精度不足 (MAPE > 30%)")

        if 'training_history' in results and results['training_history']['overfitting_ratio'] > 1.5:
            recommendations.append("⚠️  检测到过拟合风险，建议：增加正则化、数据增强、早停")
        elif 'training_history' in results and results['training_history']['overfitting_ratio'] < 1.1:
            recommendations.append("⚠️  可能欠拟合，建议：增加模型复杂度、延长训练时间")

        # 打印建议
        print("\n📝 评估建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

        # 最终评级
        print(f"\n🎯 模型总体评级:")
        good_count = sum(1 for rec in recommendations if '✅' in rec)
        warn_count = sum(1 for rec in recommendations if '⚠️' in rec)
        bad_count = sum(1 for rec in recommendations if '❌' in rec)

        total_count = len(recommendations)

        if bad_count > 0:
            print("   ❌ 需要重大改进")
        elif warn_count > good_count:
            print("   ⚠️  需要优化改进")
        elif good_count >= total_count * 0.7:
            print("   ✅ 性能良好，可以考虑部署")
        else:
            print("   ⚠️  性能一般，建议优化")

        print("\n" + "=" * 80)
        print("📁 评估完成！所有结果已保存")
        print("=" * 80)

        results['recommendations'] = recommendations
        self.results = results

        return results


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """
    使用示例
    """
    # 假设您已经有了训练好的模型和训练器
    # model = YourTrainedModel()
    # trainer = ModelTrainer(model)
    # trainer.train(...)  # 已经训练完成

    # 创建评估器
    # evaluator = ModelEvaluator(model, trainer)

    # 生成完整报告
    # report = evaluator.generate_report(
    #     X_train=X_train_tensor,  # 训练数据
    #     y_train=y_train_tensor,  # 训练标签
    #     X_val=X_val_tensor,      # 验证数据
    #     y_val=y_val_tensor,      # 验证标签
    #     X_test=X_test_tensor,    # 测试数据（可选）
    #     y_test=y_test_tensor,    # 测试标签（可选）
    #     feature_names=feature_names  # 特征名称（可选）
    # )

    print("使用方法：")
    print("1. 训练完成后，创建ModelEvaluator对象")
    print("2. 调用generate_report()方法生成完整报告")
    print("3. 查看控制台输出和可视化图表")


if __name__ == "__main__":
    example_usage()