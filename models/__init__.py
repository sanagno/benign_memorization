from torch import nn
import torchvision
from .torchvision_models import resnet18, resnet50
from .model_utils import get_normalization
from .resnet_hacks import modify_resnet_model, modify_resnet_model_medium


def get_model(
    name,
    pretrained=False,
    normalization="bn2d",
):
    models = {
        "resnet18": resnet18(
            pretrained=pretrained,
            norm_layer=get_normalization(normalization, partial=True, affine=True),
        ),
        "resnet50": resnet50(
            pretrained=pretrained,
            norm_layer=get_normalization(normalization, partial=True, affine=True),
        ),
        "resnet18_small": modify_resnet_model(
            resnet18(
                pretrained=False,
                norm_layer=get_normalization(normalization, partial=True, affine=True),
            )
        ),
        "resnet18_medium": modify_resnet_model_medium(
            resnet18(
                pretrained=False,
                norm_layer=get_normalization(normalization, partial=True, affine=True),
            )
        ),
    }

    if name not in models.keys():
        raise KeyError(f"{name} is not a valid ResNet version")

    return models[name]


class WholeModel(nn.Module):
    def __init__(
        self,
        encoder,
        projection_dim,
        projection_str=None,
        bottleneck_dim=None,
        normalization="none",
        last_normalization=None,
        final_affine=False,
    ):
        replace_fc_with_identity = (True,)
        super().__init__()

        self.encoder = encoder

        if projection_str is not None:
            self.projector = eval(projection_str)
        else:
            if bottleneck_dim is None:
                bottleneck_dim = projection_dim

            if last_normalization is None:
                last_normalization = normalization

            self.n_features = encoder.fc.in_features

            # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, bottleneck_dim, bias=False),
                get_normalization(
                    normalization,
                    dim=bottleneck_dim,
                    affine=True,
                    num_groups=bottleneck_dim // 8,
                ),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, projection_dim, bias=False),
                get_normalization(
                    last_normalization,
                    dim=projection_dim,
                    affine=final_affine,
                    num_groups=projection_dim // 8,
                ),
            )

        self.encoder.fc = nn.Identity()

    def forward(self, *args):
        h = [self.encoder(x) for x in args]
        z = [self.projector(h_i) for h_i in h]

        return *h, *z
