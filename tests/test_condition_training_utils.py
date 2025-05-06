from lerobot.common.utils.condition_training_utils import get_click_coordinates


def test_get_click_coordinates():
    import torch

    # ダミー画像 (C, H, W) = (3, 200, 300)
    img = torch.rand(3, 480, 640)  # 0–1 で生成

    x, y = get_click_coordinates(img)
    print(x,y)
    return True