import cv2
import numpy as np
from typing import Tuple, Union

def get_click_coordinates(
        img_tensor: Union["torch.Tensor", np.ndarray],
        window_name: str = "Image"
) -> Tuple[int, int]:
    """
    画像を表示し、マウス左クリックされた画素位置を返す。
    原点は画像左上。(x, y) 形式で返却。

    Parameters
    ----------
    img_tensor : torch.Tensor | np.ndarray
        ・(C, H, W), (H, W, C) または (H, W)
        ・dtype: uint8 / float32 / float64（0–1 または 0–255）

    window_name : str
        表示ウィンドウ名。

    Returns
    -------
    (int, int)
        クリック座標 (x, y)

    Raises
    ------
    RuntimeError
        クリック前に ESC キー押下またはウィンドウを閉じた場合。
    """
    # ---------- 1) Tensor → NumPy かつ 完全コピー ----------
    if "torch" in str(type(img_tensor)):        # 遅延 import 可
        import torch
        # CPU へ移しつつ clone() してメモリ共有を断つ
        img_np = img_tensor.detach().cpu().clone().numpy()
    else:
        # np.asarray → copy() で共有を断つ
        img_np = np.asarray(img_tensor).copy()

    # ---------- 2) (H, W, C) 形状へ ----------
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):  # (C, H, W)
        img_np = np.transpose(img_np, (1, 2, 0))
    elif img_np.ndim == 2:
        pass
    elif img_np.ndim != 3:
        raise ValueError("Unsupported tensor shape.")

    # ---------- 3) uint8 化 ----------
    if img_np.dtype != np.uint8:
        if np.issubdtype(img_np.dtype, np.floating) and img_np.max() <= 1.0:
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.clip(0, 255).astype(np.uint8)

    # ---------- 4) 色空間を BGR ----------
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

    # ---------- 5) コールバック設定 ----------
    global click
    click = {"pt": None}

    def _on_mouse(event, x, y, _flags, _param):
        global click
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)
            print(f"x={x}, y={y}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)
    cv2.imshow(window_name, img_np)
    cv2.waitKey(1)
    # ---------- 6) クリック待ち (ブロック) ----------
    while click["pt"] is None:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
    cv2.setMouseCallback(window_name, lambda *args: None)
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)

    cv2.waitKey(1)

    if click["pt"] is None:
        raise RuntimeError("クリックせずに終了しました。")
    return click["pt"]
