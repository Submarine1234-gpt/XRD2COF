from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
import numpy
import matplotlib.pyplot
import time

class MyXRDCalculator:
    def __init__(self, wavelength="CuKa",cifpath=None,grid=[2,30,554],fwhm=0.15,zero_shift=0.0,random_intensity=0,background=[20,25],noise=[0.002,0.008]):
        self.wavelength = wavelength
        self.cifpath = cifpath
        self.grid = grid
        self.fwhm = fwhm
        self.zero_shift = zero_shift
        self.random_intensity = random_intensity
        self.background = background
        self.noise = noise
        self.xrd_calculator = XRDCalculator(wavelength=self.wavelength)

        self.sigma = self.fwhm / (2 * numpy.sqrt(2 * numpy.log(2)))
        self.gamma = self.fwhm / 2
        self.eta = 0.4
        #使用高斯和洛伦兹混合函数模拟展宽峰形，eta为高斯和洛伦兹的混合系数

    def calculate_xrd(self) :
        origin_structure = Structure.from_file(self.cifpath)
       
        xrd_pattern =self.xrd_calculator.get_pattern(origin_structure,two_theta_range=(self.grid[0] - 2, self.grid[1] + 2))
       
        #只生成在2theta范围内的衍射峰
        two_theta_grid = numpy.linspace(self.grid[0], self.grid[1], self.grid[2])
        #创建2-30度554个点的2theta网络
        xrd_spectrum = numpy.zeros_like(two_theta_grid)
        #创建强度为0的数组
        shifted_angle = numpy.random.uniform(-self.zero_shift, self.zero_shift)
        #零点偏移

        angle = (numpy.array(xrd_pattern.x) + shifted_angle)[:,None]
        intensity = (numpy.array(xrd_pattern.y) * numpy.random.uniform(1 - self.random_intensity, 1 + self.random_intensity, len(xrd_pattern.y)))[:, None]
        delta = (two_theta_grid[None,:] - angle)
        gauss = numpy.exp(-(delta ** 2)/(2*self.sigma**2))
        lorentz = 1.0 / (1.0 + (delta / self.gamma) ** 2)
        xrd_spectrum = intensity * ((1-self.eta)*gauss + self.eta*lorentz)
        xrd_spectrum = numpy.sum(xrd_spectrum, axis=0)
        
        # for循环太慢了,改用上述矩阵
        #for angle, intensity in zip(xrd_pattern.x, xrd_pattern.y):
        #    angle = angle + shifted_angle
        #    intensity = intensity*numpy.random.uniform(1-self.random_intensity, 1+self.random_intensity)
            #随机强度

        #    gauss = numpy.exp(-((two_theta_grid - angle) ** 2) / (2 * self.sigma**2))
            #lorentz = 1.0 / (1.0 + ((two_theta_grid - angle) / self.gamma) ** 2)
            #xrd_spectrum = xrd_spectrum + intensity * ((1 - self.eta) * gauss + self.eta * lorentz)


            #xrd_spectrum = intensity * numpy.exp(-((two_theta_grid - angle) ** 2) / (2 * self.sigma**2)) + xrd_spectrum

        #计算展宽后的连续XRD谱图  
        max_peak = numpy.max(xrd_spectrum) if numpy.max(xrd_spectrum) > 0 else 1

        background_center = numpy.random.uniform(self.background[0],self.background[1])
        background_width = numpy.random.uniform(3,5)
        background_amplitude = numpy.random.uniform(0.02,0.15)*(max_peak)
        background = background_amplitude * numpy.exp(-((two_theta_grid - background_center)**2) / (2 * background_width**2))
        background_line = numpy.random.uniform(0.06,0.12)*(max_peak)
        background = background + background_line*numpy.exp(-numpy.random.uniform(0.8,1.4)*(two_theta_grid-self.grid[0]))
        #基线
        xrd_spectrum = xrd_spectrum + background
        #模拟末端大谷包

        noise = numpy.random.uniform(self.noise[0],self.noise[1])
        white_noise = numpy.random.normal(0,noise*max_peak,size=xrd_spectrum.shape)
        xrd_spectrum = xrd_spectrum + white_noise
        #白噪

        xrd_spectrum = numpy.clip(xrd_spectrum, 0, None)
        #将负值裁剪为0


        #没必要用max-min归一化
        #max_val = numpy.max(xrd_pattern.y)
        #min_val = numpy.min(xrd_pattern.y)
        ##d = max_val - min_val if max_val - min_val > 0 else 1
        #xrd_pattern.y = (xrd_pattern.y - min_val) / d
        xrd_spectrum = xrd_spectrum / numpy.max(xrd_spectrum)
        xrd_pattern.y = xrd_pattern.y / numpy.max(xrd_pattern.y)
        #归一化



        return two_theta_grid, xrd_spectrum, xrd_pattern



    
        


if __name__ == "__main__":
    time1 = time.time()
    cif = "all_cifs/88.cif"
    a = MyXRDCalculator(cifpath=cif,fwhm=0.5,zero_shift=0.1,random_intensity=0.15)
    two_theta_grid, xrd_spectrum, xrd_pattern = a.calculate_xrd()
    time2 = time.time()
    print("Time: ", time2 - time1)
    matplotlib.pyplot.plot(two_theta_grid, xrd_spectrum, label="Simulated Spectrum")
    matplotlib.pyplot.vlines(xrd_pattern.x, 0, xrd_pattern.y, colors='red', label="Bragg Peaks")
    matplotlib.pyplot.xlabel(r"2$\theta$ (degrees)", fontsize=11)
    matplotlib.pyplot.ylabel("Intensity (a.u.)", fontsize=11)
    matplotlib.pyplot.title("XRD Spectrum & Pattern Overlay", fontsize=13)
    matplotlib.pyplot.xlim(two_theta_grid.min(), two_theta_grid.max())
    matplotlib.pyplot.legend(loc="upper right")
    matplotlib.pyplot.grid(True, linestyle="--", alpha=0.5)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()