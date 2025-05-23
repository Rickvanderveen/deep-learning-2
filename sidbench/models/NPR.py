import torch.nn as nn 
import torch

from networks.resnet_npr import resnet50


class NPR(nn.Module):
    def __init__(self):
        super(NPR, self).__init__()
        self.model = resnet50(num_classes=1)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        try:
            self.model.load_state_dict(state_dict['model'], strict=False)
        except:
            print('Loading failed, trying to load model without module')
            self.model.load_state_dict(state_dict)

    def predict(self, img):
        with torch.no_grad():
            return self.forward(img).sigmoid().flatten().tolist()
        