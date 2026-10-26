# 保持空间群的 UFF 几何优化

```bash
conda activate XRD2COF
python fakeforcite.py
```

默认读取 [example_10_prepared.cif](Data/example_10_prepared.cif) 和同名
[UFF 模型](Data/example_10_prepared.uff.json)，采用 GULP UFF＋BFGS，
同时优化原子和晶胞。输出为 [example_10_optimized.cif](Data/example_10_optimized.cif)，
日志为 [example_10_optimized.log](Data/example_10_optimized.log)。

默认力阈值为 `0.002 eV/Å`，应力阈值为 `0.05 GPa`。
保留初始对称性，检查收敛和芳香键长，再重新读取导出的 CIF 核对空间群、组成。
检查失败时不会覆盖已有结果。

## 自定义输入与参数

```bash
python fakeforcite.py input.cif --uff-model input.uff.json --output optimized.cif
python fakeforcite.py --fmax 0.001 --smax 0.02 --steps 2000
```

- `--fixed-cell`：只优化原子，固定晶胞。
- `--gulp /opt/gulp-6.4/Src/gulp`：指定 GULP 可执行程序。
- `--uff-library /opt/gulp-6.4/Libraries/uff.lib`：指定势函数库。
- `--workdir gulp_debug`：保留 GULP 中间文件，便于排错；默认使用自动清理的临时目录。
- `--symprec`：对称性判断容差，默认 `0.01 Å`。
- `--logfile`：指定优化日志，默认与输出 CIF 同名。

## 通用自动补氢

[addhydrogen.py](addhydrogen.py) 保留简单的类和方法写法，通过 Open Babel
从局部几何推断键级，考虑跨晶胞邻居，并按输入的实际对称操作生成氢原子：

```python
from addhydrogen import addhydrogen

b = addhydrogen(symprec=0.01)
structure = b.run("input.cif", "input_hydrogen.cif")
```

也可以使用 `b = addhydrogen(ase_atoms)`，再调用 `structure = b.add_hydrogen()`。
返回 ASE Atoms，同时更新 `b.structure`；重原子坐标和晶胞保持原值。
默认 `keep_symmetry=True`，检查补氢前后及导出后的空间群。无法保持时会报错，
不替换输出文件。`keep_symmetry=False` 可允许补氢改变对称性。
空间群以相同 `symprec` 下的实际几何为准；CIF 标称不一致时会给出提示。

适用于满占位、中性、有合理初始几何的有机模型，使用四层成键邻居提供局部环境。
该方法仍是启发式推断，不能保证所有键级、互变异构体或质子化状态都正确。
它不生成 UFF 类型或 `.uff.json`；输出 CIF 不能直接取代优化所需的完整力场模型。
原始 [example_10.cif](Data/example_10.cif) 的共线连接，以及去氢后的
[example_10_prepared.cif](Data/example_10_prepared.cif) 的近线性亚胺，都会触发键级歧义检查。
直接运行脚本的演示使用几何合理且已有氢的
[example_10_optimized.cif](Data/example_10_optimized.cif)，输出同目录的 `example_10_hydrogen.cif`。
已验证将这个优化模型去氢后，可重新补回 8 个 H，得到 `C16H8N4`、空间群 10；
再次运行不重复加氢。金属、部分/混合占位和无法满足对称性的模型需要另行准备。

## 自动分配 UFF 类型

[settingUFF.py](settingUFF.py) 调用 Open Babel 的 UFF 分型规则，复用补氢脚本的
周期邻居环境，将类型、芳香/单/双/三/酰胺键及其晶胞偏移映射回原结构。
它只准备模型，保持原子数量、坐标、晶胞和顺序；缺氢时会提示先补氢。

```python
from settingUFF import settingUFF

b = settingUFF()
structure = b.run("input_hydrogen.cif")
# 生成 input_hydrogen.uff.json，structure.arrays["uff_types"] 保存逐原子类型。
# b.model 保存完整的 symbols、types、bonds、charges 和文件校验信息。
```

也可以用 `b = settingUFF(ase_atoms)`，再调用 `b.setUFF()` 或 `b.run()`，
仅进行内存中的分型。保存与优化入口兼容的 JSON 时使用 `run(input_file)`，
以绑定输入 CIF 的原子顺序和 SHA-256。可通过第二个参数指定 JSON 输出路径。

当前例子的完整调用方式如下；已有氢的模型可以直接从分型开始：

```bash
conda activate XRD2COF
python addhydrogen.py
python settingUFF.py
python fakeforcite.py Data/example_10_hydrogen.cif --gulp /opt/gulp-6.4/Src/gulp
```

默认沿用零电荷方案，不运行 QEq；可在构造函数中传入按原子顺序排列的 `charges`，
且总电荷必须为零。可选 `uff_library="/opt/gulp-6.4/Libraries/uff.lib"` 会提前
检查库中是否存在生成的类型；优化时 GULP 接口还会检查参数库。
自动分型仍依赖合理的几何和正确成键；键级歧义、周期像分型冲突及对称等价原子
类型不一致时会报错。含金属、部分/混合占位等模型不在当前自动流程的范围内。

## UFF 模型格式

UFF JSON 的 `source_sha256` 对应输入 CIF 的 SHA-256；`symbols`、`types`
按 ASE 展开整个晶胞后的原子顺序排列，`charges` 可省略（默认零电荷）。
`bonds` 每根键只列一次，例如：

```json
{"i": 0, "j": 1, "order": "resonant", "image": [-1, 0, 0]}
```

索引从 0 开始，键向量是 `positions[j] + image @ cell - positions[i]`。
支持 `single`、`double`、`triple`、`resonant`、`amide`。

## 验证

```bash
GULP_EXECUTABLE=/opt/gulp-6.4/Src/gulp python -m unittest discover -s test -p test_gulp_uff.py
python -m unittest discover -s test -p test_addhydrogen.py
GULP_EXECUTABLE=/opt/gulp-6.4/Src/gulp python -m unittest discover -s test -p test_settingUFF.py
```

验证力与应力有限差分、周期成键、空间群及导出、应力收敛和异常处理。
本例充分松弛后仍为 10 号，beta 约 110.73°，c 约 3.67 Å，环内 C–C 为
1.402–1.415 Å。它复现了倾斜和起伏行为，不等同于已完整复现 Materials Studio 的实现。
