"""给检测、关键点模型接上在 GPU 上做的输入转换和输出筛选（启动时在内存里改图，模型文件不动）。

为什么：分析第 1 遍里，CPU 每帧要把 640×640 的图转成 float、换维度顺序（1.3 ms），再把 4.9 MB 拷到显卡；
推理完把 8400×85 的整张输出（2.9 MB）拷回来，在 CPU 上逐行算分数。现在：
- 输入：直接送 uint8、HWC 排列的图（1.2 MB），图里先 Cast 成 float、再 Transpose，和原来 CPU 上做的是同样的精确运算，
  模型拿到的输入逐位相同。
- 输出（只有检测）：图里先按"人或球的分数 ≥ SEL_THR"挑出候选行和它们的下标，只拷回这些行；
  SEL_THR 比分析用的 0.05 略低，挑出来的是原来那批的超集，CPU 上照原样（同样的 numpy 公式）解码、按 0.05 过滤、NMS，结果逐位相同。
- 关键点：图里做 BGR→RGB、减均值、除方差（和原来 numpy 的 float32 运算顺序一致）。
用 tools/debug/check_onnx_io.py 逐帧核对过（见 docs/performance.md）。需要 onnx 包；没有就退回原来的 CPU 路径。
"""

from __future__ import annotations

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # 没装 onnx：调用方退回原路径
    onnx = None

SEL_THR = 0.045  # 输出筛选的门槛，比分析用的 0.05 低，保证是超集
U8_INPUT = "images_u8"
SEL_ROWS, SEL_IDX = "sel_rows", "sel_idx"


def _const(name: str, arr: np.ndarray):
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(arr, name + "_v"))


def _replace_input(g, new_input, nodes):
    """删掉原输入，换成 new_input；nodes 把新输入转换成原输入的名字，插在图最前面。"""
    old = g.input[0]
    keep = [i for i in g.input if i.name != old.name]
    del g.input[:]
    g.input.extend([new_input, *keep])
    for n in reversed(nodes):
        g.node.insert(0, n)
    return old.name


def yolox_u8(path: str, classes=(0, 32)) -> bytes | None:
    """YOLOX：输入 uint8 [1,H,W,3]；新增输出 sel_rows [K,85]（原始输出的候选行）和 sel_idx [K]（行号，int64）。原输出 output 保留。"""
    if onnx is None:
        return None
    m = onnx.load(path)
    g = m.graph
    dims = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]  # [1,3,H,W]
    h, w = dims[2], dims[3]
    new_in = helper.make_tensor_value_info(U8_INPUT, TensorProto.UINT8, [1, h, w, 3])
    orig_in = g.input[0].name
    _replace_input(g, new_in, [
        helper.make_node("Cast", [U8_INPUT], ["_u8_f"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["_u8_f"], [orig_in], perm=[0, 3, 1, 2]),
    ])
    out = g.output[0].name
    n_out = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim][2]  # 85
    nodes = [
        _const("_sel_shape", np.array([-1, n_out], np.int64)),
        helper.make_node("Reshape", [out, "_sel_shape"], ["_sel_o"]),  # [N,85]
        _const("_sel_obj_i", np.array([4], np.int64)),
        helper.make_node("Gather", ["_sel_o", "_sel_obj_i"], ["_sel_obj"], axis=1),  # [N,1]
        _const("_sel_thr", np.array(SEL_THR, np.float32)),
        _const("_sel_flat", np.array([-1], np.int64)),
    ]
    masks = []
    for c in classes:
        nodes += [
            _const(f"_sel_c{c}_i", np.array([5 + c], np.int64)),
            helper.make_node("Gather", ["_sel_o", f"_sel_c{c}_i"], [f"_sel_c{c}"], axis=1),
            helper.make_node("Mul", ["_sel_obj", f"_sel_c{c}"], [f"_sel_s{c}"]),
            helper.make_node("Less", [f"_sel_s{c}", "_sel_thr"], [f"_sel_lt{c}"]),
            helper.make_node("Not", [f"_sel_lt{c}"], [f"_sel_ge{c}"]),
        ]
        masks.append(f"_sel_ge{c}")
    m_any = masks[0]
    for k, mk in enumerate(masks[1:]):
        nodes.append(helper.make_node("Or", [m_any, mk], [f"_sel_or{k}"]))
        m_any = f"_sel_or{k}"
    nodes += [
        helper.make_node("Reshape", [m_any, "_sel_flat"], ["_sel_mask"]),  # [N]
        helper.make_node("NonZero", ["_sel_mask"], ["_sel_nz"]),  # [1,K]
        helper.make_node("Reshape", ["_sel_nz", "_sel_flat"], [SEL_IDX]),  # [K]
        helper.make_node("Gather", ["_sel_o", SEL_IDX], [SEL_ROWS], axis=0),  # [K,85]
    ]
    g.node.extend(nodes)
    g.output.extend([
        helper.make_tensor_value_info(SEL_ROWS, TensorProto.FLOAT, ["K", n_out]),
        helper.make_tensor_value_info(SEL_IDX, TensorProto.INT64, ["K"]),
    ])
    onnx.checker.check_model(m)
    return m.SerializeToString()


def rtmpose_u8(path: str, mean: np.ndarray, std: np.ndarray) -> bytes | None:
    """RTMPose：输入 uint8 BGR [N,H,W,3]；图里转 RGB、(x - mean) / std、换成 [N,3,H,W]。"""
    if onnx is None:
        return None
    m = onnx.load(path)
    g = m.graph
    dims = g.input[0].type.tensor_type.shape.dim  # [batch,3,H,W]
    h, w = dims[2].dim_value, dims[3].dim_value
    new_in = helper.make_tensor_value_info(U8_INPUT, TensorProto.UINT8, ["batch", h, w, 3])
    orig_in = g.input[0].name
    _replace_input(g, new_in, [
        helper.make_node("Cast", [U8_INPUT], ["_p_f"], to=TensorProto.FLOAT),
        _const("_p_bgr2rgb", np.array([2, 1, 0], np.int64)),
        helper.make_node("Gather", ["_p_f", "_p_bgr2rgb"], ["_p_rgb"], axis=3),
        _const("_p_mean", mean.astype(np.float32).reshape(1, 1, 1, 3)),
        _const("_p_std", std.astype(np.float32).reshape(1, 1, 1, 3)),
        helper.make_node("Sub", ["_p_rgb", "_p_mean"], ["_p_sub"]),
        helper.make_node("Div", ["_p_sub", "_p_std"], ["_p_norm"]),
        helper.make_node("Transpose", ["_p_norm"], [orig_in], perm=[0, 3, 1, 2]),
    ])
    onnx.checker.check_model(m)
    return m.SerializeToString()
