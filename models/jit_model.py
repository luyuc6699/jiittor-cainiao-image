import jittor as jt
import jittor.nn as nn
import numpy as np
from .jimm import tf_efficientnetv2_s_in21k


class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes, pretrain, drop_prob_list=None):
        super().__init__()

        self.base_net = tf_efficientnetv2_s_in21k(num_classes=num_classes, pretrained=pretrain)

        self.mid_features = 1280 + 160

        self.neck = nn.BatchNorm1d(self.mid_features)

        if drop_prob_list is None:
            drop_prob_list = np.linspace(0.1, 0.5, 5)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in drop_prob_list])

        self.head = nn.Linear(self.mid_features, num_classes)

    def execute(self, x):
        x = self.base_net.conv_stem(x)
        x = self.base_net.bn1(x)
        x = self.base_net.act1(x)

        second_last_block_output = None
 
        for i, block in enumerate(self.base_net.blocks):
            x = block(x)
            if i == len(self.base_net.blocks) - 2:
                second_last_block_output = x
        x = self.base_net.conv_head(x)
        x = self.base_net.bn2(x)
        x = self.base_net.act2(x)
        
        pooled_last = self.base_net.global_pool(x).flatten(1)
        pooled_last_block_output = self.base_net.global_pool(second_last_block_output).flatten(1)

        h = jt.concat([pooled_last_block_output, pooled_last], dim=1)
        h = self.neck(h)

        logit = sum([self.head(dropout(h)) for dropout in self.dropouts]) / len(self.dropouts)

        return logit

    
if __name__ == "__main__":
    model = CustomEfficientNet(num_classes=6, pretrain=False)
    x = jt.randn((4, 3, 448, 448))
    logits = model(x)
    print("output shape:", logits.shape)  