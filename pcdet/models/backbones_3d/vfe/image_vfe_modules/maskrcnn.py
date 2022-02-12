import torch 
import torch.nn as nn
import torchvision
from pathlib import Path
from torchvision.models.detection._utils import overwrite_eps
# maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

class HookedMaskRCNN(nn.Module):
    def __init__(self, network, output_layers, *args):
        super().__init__(*args)
        assert network in ['maskrcnn_resnet50_fpn']
        self.output_layers = output_layers
        self.extracted_feats = {}
        self.model_checkpoints = {
            'maskrcnn_resnet50_fpn': '../checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        # get model from torchvision
        self.pretrained = getattr(torchvision.models.detection.mask_rcnn, network)(pretrained=False)
        # load checkpoint
        self.checkpoint_path = Path(self.model_checkpoints[network])
        if not self.checkpoint_path.exists():
            ckpt = self.checkpoint_path.name
            ckpt_dir = self.checkpoint_path.parent
            ckpt_dir.mkdir(parents=True)
            url = f'https://download.pytorch.org/models/{ckpt}'
            torch.hub.load_state_dict_from_url(url, ckpt_dir)

        # self.pretrained = torch.hub.load('pytorch/vision:v0.10.0', network, pretrained=True)
        # load state dict
        self.pretrained_state_dict = torch.load(self.model_checkpoints[network]) 
        self.pretrained.load_state_dict(self.pretrained_state_dict)
        # overwrite_eps(self.pretrained, 0.0)

        self.fhooks = []

        for i, layer in enumerate(list(self.pretrained.backbone.body._modules.keys())):
            if layer in output_layers:
                self.fhooks.append(getattr(self.pretrained.backbone.body, layer).register_forward_hook(self.forward_hook(layer)))

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.extracted_feats[layer_name] = output
        return hook
    
    def forward(self, x):
        # x, _ = self.pretrained.transform(x) # GeneralizedRCNNTransform
        out = self.pretrained.backbone.body(x) # BackboneWithFPN
        return out, self.extracted_feats

