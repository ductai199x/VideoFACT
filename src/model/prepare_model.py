import importlib
from .videofact_pl_wrapper import VideoFACTPLWrapper

available_ablations_codenames = [
    "1_templates",  # 0
    "10_templates",  # 1
    "concat_scaled_fe",  # 2
    "no_cfe",  # 3
    "no_pfe",  # 4
    "no_scaling_no_combine_fe_embed",  # 5
    "no_transformer",  # 6
    "no_transformer_yes_templates",  # 7
]


def prepare_model(ablation_codename: str, prev_ckpt: str, config):
    if ablation_codename == "":
        import_path = f"src.model.common.videofact"
    else:
        if ablation_codename not in available_ablations_codenames:
            raise NotImplementedError(
                f"{ablation_codename} does not exists. Available codenames for v11 are: {available_ablations_codenames}"
            )
        import_path = f"src.model.ablations.videofact_{ablation_codename}"
    torch_model = importlib.import_module(import_path).VideoFACT

    print("\n" + "#" * 40 + "\n" + f"LOADING MODEL: {import_path}" + "\n" + "#" * 40 + "\n")
    if prev_ckpt:
        return VideoFACTPLWrapper.load_from_checkpoint(prev_ckpt, model=torch_model, **config)
    else:
        return VideoFACTPLWrapper(model=torch_model, **config)
