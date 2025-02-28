from .clip import clip 
from PIL import Image
import torch.nn as nn
from .clip.model import Projector
import torch


CHANNELS = {
    "RN50" : 1024,
    # "ViT-L/14" : 1024
    "ViT-L/14" : 768
}


def dotproduct_similarity(x, y):
    return torch.sum(x * y, dim=1)

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.model.float()

        # detector, we adopt the weight as the authenticity prototype
        self.fc = nn.Linear(CHANNELS[name], num_classes )

        # forgery feature projector
        self.forgery_proj = Projector(CHANNELS[name], 'fc')
        # authenticity feature projector
        self.authenticity_proj = Projector(1, 'fc')

    def forward(self, x, return_feature=False):
        forgery_features, semantic_features = self.model.encode_image(x) 

        output = self.fc(forgery_features)
        
        # authenticity of forgery features(dot product between forgery features and the authenticity prototype)
        authenticity = dotproduct_similarity(forgery_features, self.fc.weight).unsqueeze(1)
        # authenticity of semantic features(dot product between semantic features and the authenticity prototype)
        authenticity_old = dotproduct_similarity(semantic_features.detach(), self.fc.weight).unsqueeze(1)

        if return_feature:
            return output, forgery_features, semantic_features
        
        # projected feature, projected authenticity, semantic features, prediction, authenticity of semantic features
        return self.forgery_proj(forgery_features), self.authenticity_proj(authenticity), semantic_features, output, authenticity_old

