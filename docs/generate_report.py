"""Generate the one-page FreeTune4D usage report as a PDF."""

from pathlib import Path


OUT_DIR = Path(__file__).parent
TITLE = "FreeTune4D 当前代码使用报告（一页版）"

SECTIONS = [
    ("1. 用途与输入输出", [
        "用途：利用同一受试者的高质量三维（UQ-3D）MRI作为解剖先验，将低质量动态四维（LQ-4D）MRI重建为具有高空间质量和呼吸运动信息的UQ-4D MRI；当前脚本面向T2加权数据。",
        "输入：动态4D MRI DICOM、静态高质量3D T2 DICOM、患者/检查日期/序列名、DICOM参考头，以及coarse.h5和fine.h5两级权重。预处理脚本按相位聚类、平均、重采样与空间对齐，生成phase_T2.mat（FourD_ave_save、T2_save）。",
        "输出：每个呼吸相位的UQ_T2_<帧>.mat及T2w_frame<帧>_<层>.dcm；同时输出对应LQ DICOM和质控图。默认最多处理0—64帧。",
    ]),
    ("2. 算法流程与特点", [
        "先以全局相关系数选择与3D T2最匹配的4D参考相位，经Elastix刚性+B样条配准完成初始对齐；随后以两级HyperVxmDense估计形变场：粗级128³、64通道，细级224³、24通道，均采用5步积分。粗级各相位形变场沿时间轴作三次B样条平滑，再由细级残差形变校正，最后用空间变换器将高质量3D T2传播到全部呼吸相位。",
        "特点：无需面向新中心进行梯度微调/再训练；融合解剖先验与动态运动；两级形变兼顾大范围运动与细节；时间平滑增强呼吸运动连续性；结果可直接导出DICOM。局限是严重伪影、极低软组织对比度或椎体近乎不可见时性能可能下降。",
    ]),
    ("3. 论文精度与速度（均值±标准差）", [
        "外部验证：机构B/T1-w的NCC、NMI、LCC分别为0.313±0.068、0.365±0.072、0.126±0.014，肝/脾运动误差2.629±1.750/4.635±2.575 mm；机构B/T2-w为0.265±0.024、0.345±0.044、0.128±0.030，误差1.576±1.345/2.371±1.178 mm；机构C/T1-w为0.305±0.048、0.388±0.051、0.072±0.018，误差4.770±2.328/2.650±1.358 mm。NCC/NMI/LCC越高越好，运动误差越低越好；上述为跨模态统计，不能表述成单一“准确率”。",
        "论文在机构A的10帧T1-w数据上报告911.2±156.3 ms/帧。论文测速工作站为AMD EPYC 9654、NVIDIA RTX A6000、256 GB内存；这是实测配置，不是最低配置。",
    ]),
    ("4. 环境、模型规模与运行要求", [
        "仓库锁定TensorFlow-GPU/Keras 2.10.0、PyTorch 2.1.0+cu121（CUDA 12.1构建）、TorchVision 0.16.0、VoxelMorph 0.2、NumPy 1.26.4、SciPy 1.13.1、SimpleITK 2.3.1及pydicom 2.4.4；另需Elastix。代码将设备固定为cuda，当前未提供CPU回退路径。TensorFlow 2.10与PyTorch cu121能否共用同一CUDA环境仍需实际验证，建议隔离环境或按已验证部署镜像执行。",
        "最低配置：仓库与论文均未给出最低GPU/显存、CPU核心数和内存，不能据A6000测试机反推；需用代表性病例逐级降低资源进行峰值显存、内存和耗时测试。参数量：仓库未包含coarse.h5/fine.h5，论文截图亦未报告，因此暂无法可靠统计；取得权重后以model.count_params()分别记录并求和。",
    ]),
]

FOOTER = "依据：当前仓库代码/requirements.txt；IEEE JBHI作者版本，DOI: 10.1109/JBHI.2026.3698133（表II、III、VI及讨论）。仅供科研/部署评估，临床使用前须完成本地验证。"


def all_lines():
    lines = [(TITLE, "title")]
    for heading, paragraphs in SECTIONS:
        lines.append((heading, "heading"))
        lines.extend(("• " + paragraph, "body") for paragraph in paragraphs)
    lines.append((FOOTER, "footer"))
    return lines


def wrap_cjk(text, width):
    lines, current = [], ""
    for char in text:
        weight = 1 if ord(char) > 127 else 0.55
        current_weight = sum(1 if ord(c) > 127 else 0.55 for c in current)
        if current and current_weight + weight > width:
            lines.append(current)
            current = char
        else:
            current += char
    if current:
        lines.append(current)
    return lines


def pdf_hex(text):
    return "FEFF" + text.encode("utf-16-be").hex().upper()


def make_pdf(path):
    commands = ["BT"]
    y = 812
    for text, style in all_lines():
        size = {"title": 16, "heading": 11.5, "body": 8.6, "footer": 7.2}[style]
        leading = {"title": 20, "heading": 15, "body": 11.1, "footer": 9}[style]
        width = {"title": 33, "heading": 66, "body": 78, "footer": 90}[style]
        if style == "title":
            x = 155
        else:
            x = 38
        if style == "heading":
            y -= 2
        for line in wrap_cjk(text, width):
            commands.extend([f"/F1 {size} Tf", f"1 0 0 1 {x} {y} Tm", f"<{pdf_hex(line)}> Tj"])
            y -= leading
        y -= 2 if style in {"title", "heading"} else 1
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [6 0 R] >>",
        b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(pdf)


if __name__ == "__main__":
    make_pdf(OUT_DIR / "FreeTune4D_当前代码使用报告.pdf")
