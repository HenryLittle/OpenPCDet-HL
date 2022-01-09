import torch 
import torch.nn as nn

# resnet18
# layer1 torch.Size([1, 64, 94, 311])
# layer2 torch.Size([1, 128, 47, 156])
# layer3 torch.Size([1, 256, 24, 78])
# layer4 torch.Size([1, 512, 12, 39])

class HookedResNet(nn.Module):
    def __init__(self, resnet, output_layers, *args):
        super().__init__(*args)
        assert resnet in ['resnet18', 'resnet34', 'resnet50']
        self.output_layers = output_layers
        self.extracted_feats = {}
    
        self.pretrained = torch.hub.load('pytorch/vision:v0.10.0', resnet, pretrained=True)
        self.pretrained.eval()
        self.fhooks = []

        for i, layer in enumerate(list(self.pretrained._modules.keys())):
            if layer in output_layers:
                self.fhooks.append(getattr(self.pretrained, layer).register_forward_hook(self.forward_hook(layer)))

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.extracted_feats[layer_name] = output
        return hook
    
    def forward(self, x):
        out = self.pretrained(x)
        return out, self.extracted_feats

