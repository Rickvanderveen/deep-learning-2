import torch
import torch.nn as nn

from sidbench.networks.clip import clip 


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class RineModel(nn.Module):
    def __init__(self, backbone, nproj, proj_dim):
        super(RineModel, self).__init__()

        # Load and freeze CLIP
        self.clip, _ = clip.load(backbone[0], device="cpu")
        for _, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module) for name, module in self.clip.visual.named_modules() if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))

        proj1_layers = [
            nn.Dropout()
        ]

        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)

        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)

        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )

    def forward(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g
        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return p, z

    def get_embedding(self, x):
        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

            g = self.proj1(g.float())

            z = torch.softmax(self.alpha, dim=1) * g
            z = torch.sum(z, dim=1)
            z = self.proj2(z)

        return z

    def predict(self, img):
        with torch.no_grad():
            logits, _ = self.forward(img)
            return logits.sigmoid().flatten().tolist()
        
    def load_weights(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        for name in state_dict:
            exec(f'self.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])')