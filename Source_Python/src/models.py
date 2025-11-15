import torch
import torch.nn as nn
import timm


class ImageMetaModel(nn.Module):
    """Wrapper: an image backbone from timm + optional MLP for metadata.

    - backbone_name: timm model name
    - num_meta_features: number of tabular features to fuse
    - out_features: final output classes (1 for binary)
    """

    def __init__(self, backbone_name: str, num_meta_features: int = 0, out_features: int = 1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        # timm sets attribute 'num_features'
        feat_dim = getattr(self.backbone, 'num_features', None)
        if feat_dim is None:
            # fallback
            feat_dim = 512

        self.meta_mlp = None
        if num_meta_features and num_meta_features > 0:
            self.meta_mlp = nn.Sequential(
                nn.Linear(num_meta_features, 64),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(64),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
            )

        combined_dim = feat_dim + (32 if self.meta_mlp is not None else 0)

        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, out_features),
        )

    def forward(self, image, meta=None):
        x = self.backbone(image)
        if self.meta_mlp is not None and meta is not None:
            m = self.meta_mlp(meta)
            x = torch.cat([x, m], dim=1)
        out = self.head(x)
        return out.squeeze(1)


def get_model(name: str, num_meta_features: int = 0, out_features: int = 1, pretrained=True):
    # Provide aliases for common names
    alias = name.lower()
    # Use timm names directly for many models. Allow convenient mapping.
    mapping = {
        'resnet50': 'resnet50',
        'resnet101': 'resnet101',
        'inceptionv3': 'inception_v3',
        'densenet121': 'densenet121',
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'efficientnet_b2': 'efficientnet_b2',
        'efficientnet_b3': 'efficientnet_b3',
        'efficientnet_b4': 'tf_efficientnet_b4',
        'efficientnet_b5': 'tf_efficientnet_b5',
        'efficientnet_b6': 'tf_efficientnet_b6',
        'efficientnet_b7': 'tf_efficientnet_b7',
        'regnety': 'regnety_032',
        'regnetx': 'regnetx_032',
        'vit_b16': 'vit_base_patch16_224',
        'vit_l16': 'vit_large_patch16_384',
        'swin': 'swin_base_patch4_window7_224',
        'coatnet': 'coatnet_0_rw_224',
    }

    chosen = mapping.get(alias, name)
    model = ImageMetaModel(chosen, num_meta_features=num_meta_features, out_features=out_features, pretrained=pretrained)
    return model
