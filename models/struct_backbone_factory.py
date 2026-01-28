import torch

from models.struct_backbone_rotate import RotatEBackbone
from models.struct_backbone_complex import ComplExBackbone


def resolve_struct_ckpt(args):
    ckpt = getattr(args, "pretrained_struct", None)
    if ckpt:
        return ckpt
    return getattr(args, "pretrained_rotate", None)


def load_struct_backbone(
    struct_type: str,
    num_entities: int,
    num_relations: int,
    emb_dim: int,
    margin: float,
    ckpt_path: str | None,
    device: torch.device,
):
    struct_type = (struct_type or "rotate").lower()
    if struct_type == "rotate":
        model = RotatEBackbone(num_entities, num_relations, emb_dim=emb_dim, margin=margin).to(device)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.model.load_state_dict(ckpt, strict=True)
    elif struct_type == "complex":
        model = ComplExBackbone(num_entities, num_relations, emb_dim=emb_dim).to(device)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=True)
    else:
        raise ValueError(f"Unknown struct_type={struct_type}")

    model.eval()
    return model
