import os
import torch
from .densenet import ImprovedDenseNet
from config.settings import WEIGHTS_PATH


def get_model(pretrained=True, weights_path=WEIGHTS_PATH):
    """Instantiate ImprovedDenseNet and load weights if available."""
    model = ImprovedDenseNet(pretrained=pretrained)
    
    if os.path.exists(weights_path):
        try:
            try:
                state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            
            # If checkpoint dictionary saved instead of state_dict directly
            if isinstance(state_dict, dict) and 'model_state' in state_dict:
                state_dict = state_dict['model_state']
                
            model.load_state_dict(state_dict)
        except Exception as e:
            pass

    model.eval()
    return model
