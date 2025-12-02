from torch import nn

class FluPredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            # 第一层（比简化模型大，比原模型小）
            nn.Linear(input_size, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.15),  # 介于0.3和0.1之间

            # 第二层
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.15),

            # 第三层
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 输出层
            nn.Linear(48, 1)
        )

    def forward(self, x):
        return self.model(x)
    # def __init__(self, input_size):
    #     super(FluPredictionModel, self).__init__()
    #
    #     self.model = nn.Sequential(
    #         nn.Linear(input_size, 256),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #
    #         nn.Linear(256, 128),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU(),
    #         nn.Dropout(0.3),
    #
    #         nn.Linear(128, 64),
    #         nn.BatchNorm1d(64),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #
    #         nn.Linear(32, 1)
    #     )
    #
    #     # 初始化权重
    #     self._init_weights()
    #
    # def _init_weights(self):
    #     """初始化权重"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #
    # def forward(self, x):
    #     return self.model(x)