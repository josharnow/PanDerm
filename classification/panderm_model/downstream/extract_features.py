import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")
# import torchsnooper
# @torchsnooper.snoop()
@torch.no_grad()
def extract_features_from_dataloader(args, model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.
    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.
    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, (2) [N x 1]-dim np.array of labels, and (3) list of filenames
    """
    all_embeddings, all_labels, all_filenames = [], [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters()).device
    for batch_idx, (batch, target, filename) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            padding = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, padding])
        batch = batch.to(device)
        with torch.inference_mode():
            """comment out it if using biomed_clip"""
            # embeddings, text_features, logit_scale=model(batch)
            # embeddings=embeddings.cpu()[:remaining, :].cpu()
            """comment out it if using MONET, virtual_env: MILAN"""
            # embeddings = model(batch)[1]
            # # print('EEEEEEEEEEEE',embeddings.size())
            # embeddings=embeddings.cpu()[:remaining, :].cpu()
            #open_clip
            # embeddings = model.encode_image(batch).cpu()[:remaining, :].cpu()
            if args.model == 'PanDerm-Large' or args.model == 'PanDerm-Base':
                embeddings = model.forward_features(batch, is_train=False).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()
        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_filenames.extend(filename[:remaining])
    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "filenames": all_filenames
    }
    return asset_dict
