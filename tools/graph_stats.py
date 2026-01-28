import torch


def load_graph_stats(path: str, device: torch.device):
    obj = torch.load(path, map_location="cpu")
    tensor_keys = [
        "nbr_ent",
        "nbr_rel",
        "nbr_dir",
        "nbr_mask",
        "freq",
        "deg",
        "rel_hist_ids",
        "rel_hist_counts",
    ]
    for key in tensor_keys:
        if key in obj and torch.is_tensor(obj[key]):
            obj[key] = obj[key].to(device)
    return obj
