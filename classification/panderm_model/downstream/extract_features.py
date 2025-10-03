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
    pbar = tqdm(total=len(dataloader), desc="Extracting features")
    for batch_idx, (batch, target, filename) in enumerate(dataloader):
        # Show the current image being processed (first filename in batch)
        try:
            if isinstance(filename, (list, tuple)) and len(filename) > 0:
                current_img = filename[0]
            else:
                current_img = str(filename)
            # Print a line for logs and update tqdm postfix for live view
            print(f"Processing image: {current_img}", flush=True)
            pbar.set_postfix_str(str(current_img))
        except Exception:
            # Don't fail the run if filename formatting is unexpected
            pass
        remaining = batch.shape[0]
        if remaining != batch_size:
            padding = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, padding])
        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model.forward_features(batch, is_train=False).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()
        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_filenames.extend(filename[:remaining])
        pbar.update(1)
    pbar.close()
    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "filenames": all_filenames
    }
    return asset_dict
