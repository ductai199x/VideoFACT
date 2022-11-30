from typing import Tuple

from .common import *


class E2fgviDavisDataset(CommonImageDataset):
    def __init__(
        self,
        dataset_samples: str,
        fixed_img_size: Tuple[int, int] = (1080, 1920),
        mask_available=True,
        return_labels=True,
        return_file_path=False,
        need_rotate=False,
        need_resize=True,
        need_crop=False,
    ):
        super().__init__(
            dataset_samples,
            fixed_img_size,
            mask_available,
            return_labels,
            return_file_path,
            need_rotate,
            need_resize,
            need_crop,
        )

    def _get_mask(self, folder, filename):
        try:
            mask = to_tensor(Image.open(f"{folder}/{filename}.mask", mode="r"))
            if mask.shape < 3:
                mask = mask.unsqueeze(0)
            mask = self._transform(mask).squeeze()
            mask[mask > 0] = 1
            mask = mask.int()
        except:
            mask = torch.zeros((1080, 1920), dtype=torch.uint8)
        return mask
