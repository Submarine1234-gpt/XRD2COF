import torch
import numpy
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
import matplotlib.pyplot as plt
def cif2xrd(cif_path):
    xrd_calc = XRDCalculator(wavelength="CuKa")
    structure = Structure.from_file(cif_path)
    xrd_pattern = xrd_calc.get_pattern(structure)

    two_theta_grid = numpy.linspace(2, 30, 554)
    xrd_spectrum = numpy.zeros_like(two_theta_grid)
    sigma = 0.15 / (2 * numpy.sqrt(2 * numpy.log(2)))
    for angle, intensity in zip(xrd_pattern.x, xrd_pattern.y):
        xrd_spectrum = intensity * numpy.exp(-((two_theta_grid - angle) ** 2) / (2 * sigma**2)) + xrd_spectrum
    return two_theta_grid, xrd_spectrum, xrd_pattern


def plot_xrd_comparison(two_theta, xrd_spectrum, xrd_pattern, mode="subplots"):
    """
    对比连续展宽图谱 (xrd_spectrum) 与离散衍射峰 (xrd_pattern)
    :param two_theta: 2theta 网格数组
    :param xrd_spectrum: 展宽后的连续 XRD 谱图
    :param xrd_pattern: pymatgen 计算得到的离散 DiffractionPattern
    :param mode: "subplots"（上下子图联动对比）或 "overlay"（同图叠加对比）
    """
    theta_min, theta_max = two_theta.min(), two_theta.max()
    # 筛选在当前 2theta 范围内的衍射峰
    mask = (xrd_pattern.x >= theta_min) & (xrd_pattern.x <= theta_max)
    peaks_x = xrd_pattern.x[mask]
    peaks_y = xrd_pattern.y[mask]

    if mode == "subplots":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        # 上子图：连续展宽图谱
        ax1.plot(two_theta, xrd_spectrum, color="#1f77b4", lw=1.8, label="Broadened Spectrum (xrd_spectrum)")
        ax1.set_ylabel("Intensity (a.u.)", fontsize=11)
        ax1.set_title("XRD Spectrum vs Diffraction Pattern", fontsize=13)
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 下子图：离散衍射峰棒状图
        ax2.vlines(peaks_x, 0, peaks_y, color="#d62728", lw=1.5, label="Bragg Peaks (xrd_pattern)")
        ax2.set_xlabel(r"2$\theta$ (degrees)", fontsize=11)
        ax2.set_ylabel("Relative Intensity (%)", fontsize=11)
        ax2.set_xlim(theta_min, theta_max)
        ax2.set_ylim(0, max(peaks_y.max() if len(peaks_y) > 0 else 100, 100) * 1.05)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

    elif mode == "overlay":
        plt.figure(figsize=(9, 5))
        plt.plot(two_theta, xrd_spectrum, color="#1f77b4", lw=1.8, label="Broadened Spectrum (xrd_spectrum)")
        plt.vlines(peaks_x, 0, peaks_y, color="#d62728", lw=1.2, alpha=0.75, label="Bragg Peaks (xrd_pattern)")
        plt.xlabel(r"2$\theta$ (degrees)", fontsize=11)
        plt.ylabel("Intensity (a.u.)", fontsize=11)
        plt.title("XRD Spectrum & Pattern Overlay", fontsize=13)
        plt.xlim(theta_min, theta_max)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cif_path = "all_cifs/5.cif"  
    two_theta, xrd_spectrum, xrd_pattern = cif2xrd(cif_path)
    numpy.savetxt("1.txt", numpy.column_stack((two_theta, xrd_spectrum)))
    
    # 绘制对比图: 可选择 "subplots" (上下双图联动) 或 "overlay" (单图叠加)
    plot_xrd_comparison(two_theta, xrd_spectrum, xrd_pattern, mode="overlay")
    