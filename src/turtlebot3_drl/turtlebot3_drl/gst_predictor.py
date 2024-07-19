import torch
from types import SimpleNamespace
from .gumbel_social_transformer import GumbelSocialTransformer
from .st_model import st_model

def load_gst_model(model_path):
    # Load the model configuration (you might need to adjust this based on your training setup)
    args = SimpleNamespace(
        spatial='gumbel_social_transformer',
        temporal='faster_lstm',
        motion_dim=2,
        output_dim=5,
        embedding_size=64,
        spatial_num_heads=8,
        spatial_num_heads_edges=0,
        spatial_num_layers=1,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        decode_style='recursive',
        detach_sample=False,
        only_observe_full_period=True,
        ghost=False,
        pred_seq_len=12,
        obs_seq_len=8,
    )

    # Initialize the model
    model = st_model(args, device='cpu')  # Force CPU usage

    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # Filter out unexpected keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    
    # Load the filtered state dict
    model.load_state_dict(model_dict, strict=False)
    model.eval()

    return model

def predict_trajectory(model, input_data, pred_steps=12):
    # Prepare input data (you might need to adjust this based on your model's input requirements)
    x = input_data.unsqueeze(0)  # Add batch dimension (input_data already has time dimension)
    A = torch.zeros(1, input_data.shape[0], input_data.shape[1], input_data.shape[1], 2)  # Dummy edge input
    attn_mask = torch.ones(1, input_data.shape[0], input_data.shape[1], input_data.shape[1])
    loss_mask_rel = torch.ones(1, input_data.shape[1], 1)

    with torch.no_grad():
        model_output = model(x, A, attn_mask, loss_mask_rel, device='cpu')
        if isinstance(model_output, tuple) and len(model_output) >= 2:
            x_sample_pred = model_output[1]
        else:
            print(f"Unexpected model output: {model_output}")
            return None

    # Extract the predicted trajectories
    predictions = x_sample_pred.squeeze(0).tolist()  # Remove batch dimension
    return predictions