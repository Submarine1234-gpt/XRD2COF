import numpy as np
import torch
from torch.utils.data import Dataset

class RobustCOFXRDDataset(Dataset):
    def __init__(self, raw_data_list, is_training=True):
        """
        raw_data_list: 每个元素包含由 pymatgen 提取的基础峰:
                       {'angles': np.array([...]), 'intensities': np.array([...]), 'label': ...}
        """
        self.data = raw_data_list
        self.is_training = is_training
        self.grid = np.linspace(5, 110, 5250)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        angles = item['angles'].copy()
        intensities = item['intensities'].copy()

        if self.is_training:
            # 1. 随机零点漂移 (±0.12°)
            angles += np.random.uniform(-0.12, 0.12)
            
            # 2. 随机 FWHM 展宽 (0.08° ~ 0.30°)
            fwhm = np.random.uniform(0.08, 0.30)
            
            # 3. 随机相对峰强缩放 (模拟择优取向)
            intensities *= np.random.uniform(0.85, 1.15, size=len(intensities))
        else:
            # 验证/测试集使用固定标准参数
            fwhm = 0.15

        # --- 生成连续谱图 ---
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        spectrum = np.zeros_like(self.grid)
        for ang, inten in zip(angles, intensities):
            if 5 <= ang <= 110:
                spectrum += inten * np.exp(-((self.grid - ang) ** 2) / (2 * sigma ** 2))

        # --- 叠加背景与底噪 (仅在训练时施加) ---
        if self.is_training:
            # 添加低角度非晶背景
            bg_amp = np.random.uniform(0.0, 0.10) * (spectrum.max() + 1e-6)
            spectrum += bg_amp * np.exp(-((self.grid - np.random.uniform(15, 25)) ** 2) / (2 * 10**2))
            
            # 添加白噪声
            noise = np.random.normal(0, 0.01 * (spectrum.max() + 1e-6), size=spectrum.shape)
            spectrum = np.clip(spectrum + noise, 0, None)

        # 归一化到 [0, 1]
        if spectrum.max() > 0:
            spectrum /= spectrum.max()

        return torch.tensor(spectrum, dtype=torch.float32), item['target_property']