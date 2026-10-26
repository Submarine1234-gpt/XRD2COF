# 最小 SQL 建胞实验

这个文件夹可以单独复制。三个 Python 文件只相互导入，不调用项目原有的
`cof/` 或 `workspace/` 代码。运行环境需要 `rdkit`、`numpy`、`scipy`、
`gemmi`、`pymatgen` 和 `spglib`；项目现有的 XRD 环境已经包含它们。

| 文件 | 职责 |
|---|---|
| `input.py` | 从两条 SMILES 自动辨认四醛与四胺，报告原始反应端编号、端口分子图对称置换、二维端口夹角；从模板 CIF 的空间群生成同晶胞子群树。 |
| `groups.py` | 读取模板 CIF，调用本机 pymatgen 的群操作数据库，按操作集合的包含关系生成树。 |
| `place.py` | 把两个单体的完整重原子图放到 CIF 的 A/B 节点，按照四条周期边接成 C=N；检查端口角、初始键长、对称操作和周期键图，再写 CIF。 |
| `sql_nodes.cif` | 可编辑的抽象节点模板：根群 #123、A/B 位置及四条带晶胞平移的 sql 边。A/B 是占位点，不是化学原子。 |

## 两步运行

在项目根目录的 PowerShell 中执行。复制此文件夹后，在它的父目录执行
相同的 `-m mini_sql_lab.*` 命令即可。

```powershell
& 'C:\Users\LocalSubmarine\miniconda3\envs\XRD\python.exe' -X utf8 -B -m mini_sql_lab.input --first-smiles 'O=Cc1cc(C=O)c(C=O)cc1C=O' --second-smiles 'Nc1cc(N)c(N)cc1N' --out mini_sql_lab/example_input.json

& 'C:\Users\LocalSubmarine\miniconda3\envs\XRD\python.exe' -X utf8 -B -m mini_sql_lab.place --input mini_sql_lab/example_input.json --group 10 --out mini_sql_lab/example_10.cif
```

第一步产生 `example_input.json`。其中的 `monomers` 区分**端口分子图对称性**与
RDKit 二维画法得到的角度；`space_group_tree` 列出模板 #123 在同一晶胞基底、
同一原点下的子群。若 SMILES 有多于四个反应端，可用 `--first-ports 1,5,8,12`
等参数选择**原始 SMILES 原子编号**。

第二步的 `--group` 必须在第一步的树里。成功时输出 `example_10.cif` 与
`example_10_report.json`；失败时只写失败报告，不产生新 CIF。报告里有
实际端口配对、四元孔道角和偏差、C=N 初始距离、对称映射残差及逐取向拒绝原因。
把 `--group` 换成 `1`、`16`、`25` 可以逐个实验。换单体时先重新运行第一步。
报告还给出 spglib 对**无氢初始重原子结构**检测到的群；它可能高于指定群，
不能当作补氢和优化后的最终最高群。随附示例声明 #10，重原子初值被检测为
#47，正是这种偶然升群情形。

## 这份最小代码的边界

- 只支持**二维单层 sql、两个各四个反应端的醛/胺单体、亚胺键**。网络 sql
  是模板预先指定的；程序不会从 SMILES 自动判定应该使用 sql、she 或其它拓扑。
- 输出是**无氢的重原子初始位置**，没有力场、构象集合或晶胞及原子坐标的
  几何优化。CIF 的群操作和周期键图经过核对，但它不是实验结构或最终 COF。
- 单体“对称性”指所选端口在分子图自同构下的置换。二维角度来自 RDKit 绘图；
  两者都不是测得的三维分子点群。
- 子群树只比较 pymatgen 标准设置中的**同晶胞、同原点**操作集合，
  不含换基、移原点、扩胞的完整晶体学子群树。群操作数据来自运行环境里的
  `pymatgen.symmetry.groups.SpaceGroup.SYMM_OPS`；版本写在输入报告中。
- `sql_nodes.cif` 的 A/B 坐标和四条边沿用本项目 `data/nets.json` 的手工
  二节点 sql 嵌入；RCSR 提供 sql 网的名称，模板的这个具体嵌入不是从
  RCSR 自动下载的。
- `--max-symmetry-rms` 默认 1.2 Å 是**初始构象筛选预算**，不是空间群定律。
  `--max-bond-change` 默认 0.25 Å 会挡住对称投影压坏单体键长的初值。
  例如带甲氧基的单体在 #10 下需要更充分的群约束优化；这个小程序会拒绝
  直接平均后严重变形的初始胞。拒绝只说明此最小搜索未找到合格初值。

建议先只改两条 SMILES、`--group` 和模板的 A/B 位置与四条 `_mini_edge_*`
记录；观察 JSON 中的端口配对和拒绝理由，再考虑增加构象或优化步骤。
