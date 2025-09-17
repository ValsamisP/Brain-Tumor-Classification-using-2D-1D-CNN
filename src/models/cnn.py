"""
CNN model with 2D+1D arcihtecture, transfer learning and attention mechanisms
"""

import torch
import torch.nn as nn
from torchvision import models 
from torchvision.models import ResNet18_Weights

class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important spatial regions"""

    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2,1,kernel_size,padding = kernel_size//2 , bias =False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim =True)
        max_out, _ =torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out,max_out],dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)

class ChannelAttention(nn.Module):
    """Channel attention mechanism to focus on important feature channels"""

    def __init__(self,in_planes,ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1 ,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class Enhanced_CNN2D1D(nn.Module):
    """
    2D + 1D CNN with transfer learning and attention mechanisms
    
    Architecture:
    1. Pre-trained ResNet18 backbone for feature extraction
    2. Additional 2D CNN layers for domain-specific features
    3. Channel and spatial attention mechanisms
    4. 1D CNN for sequential processing
    5. Skip connections and advanced regularization
    """

    def __init__(self,num_classes=4, pretrained=True):
        super(Enhanced_CNN2D1D,self).__init__()

        #Pre-trained backbone 
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Additional 2D CNN layers for domain adaptation
        self.conv2d_enhance = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(256,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )
    

        # Attention mechanisms
        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # 1D CNN for sequential processing of features
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(256,512,kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv1d(512,256,kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        # Classification head with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 512), #Skip connection from attention
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if m  not in self.backbone.modules():
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m not in self.backbone.modules():
                    nn.init.constant_(m.weight,1)
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        #Extract features using pre-trained backbone
        x = self.backbone(x) # [batch_size,512,H,W]

        #Enhance features with domain-specific layers
        x = self.conv2d_enhance(x) # [batch_size,128,H,W]

        # Apply attention mechanisms
        channel_att = self.channel_attention(x)
        x = x * channel_att

        spatial_att = self.spatial_attention(x)
        x = x * spatial_att

        # Create skip conncection
        skip_features = self.global_avg_pool(x).view(x.size(0), -1) # [batch_size,128]

        # Prepare for 1D CNN
        x = self.global_avg_pool(x) # [batch_size,128,1,1]
        x = x.view(x.size(0), x.size(1), 1) #[batch_size,128,1]

        # 1D CNN processing
        x = self.conv1d_block(x) #[batch_size,256,1]

        # Flatten for classification
        x = x.view(x.size(0), -1) # [batch_Size,256]

        # Combine with skip connection
        x = torch.cat([x, skip_features],dim=1) #[batch_size,384]

        # Final classification 
        x = self.classifier(x)

        return x

    def get_features_maps(self, x, layer_name='attention'):
        """Extract features maps for visualization"""

        features = {}

        # Extract backbone features

        x = self.backbone(x)
        features['backbone'] = x

        # Enhanced features 
        x = self.conv2d_enhance(x)
        features['enhanced'] = x

        # Attention maps 
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        features['channel_attention'] = channel_att
        features['spatial_attention'] = spatial_att 

        # Attended features 
        x = x * channel_att * spatial_att
        features['attended'] = x 

        return features     