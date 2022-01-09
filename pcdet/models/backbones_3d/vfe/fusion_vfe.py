import torch

from .vfe_template import VFETemplate
from .image_vfe import ImageVFE
from .mean_vfe import MeanVFE
from .image_vfe_modules.resnet import HookedResNet

class ImageResNetVFE(VFETemplate):
    def __init__(self, model_cfg, point_cloud_range, num_point_features, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.num_point_features = num_point_features
        # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.pc_range = point_cloud_range 

        self.resnet = HookedResNet(resnet=model_cfg.BACKBONE,
        output_layers=model_cfg.OUTPUT_LAYERS)

        self.mean_vfe = MeanVFE(model_cfg, num_point_features)
    
    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):

        # just get feature pyramid shape {key:[B, C, H, W]}
        batch_dict['images'] = torch.nan_to_num(batch_dict['images'])
        with torch.no_grad():
            _, image_fpn = self.resnet(batch_dict['images'])

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict['image_fpn'] = image_fpn

        return batch_dict