from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


source = next(Path('D:/huihui').glob('*/*/面向遮挡场景的可见性感知多视图深度估计方法.docx'))
original_hash = sha256(source.read_bytes()).hexdigest()
output = Path('outputs/manuscript/面向遮挡场景的可见性感知多视图深度估计方法（机制验证修订稿）.docx')
output.parent.mkdir(parents=True, exist_ok=True)
doc = Document(source)
p = doc.paragraphs
tables = doc.tables


def replace(container, old, new):
    nodes = container.xpath('.//w:t')
    text = ''.join(node.text or '' for node in nodes)
    for start in reversed([i for i in range(len(text)) if text.startswith(old, i)]):
        end = start + len(old)
        pos = 0
        for node in nodes:
            value = node.text or ''
            stop = pos + len(value)
            if pos < end and stop > start:
                node.text = value[:max(start-pos, 0)] + (new if pos <= start else '') + value[max(end-pos, 0):]
                node.set(qn('xml:space'), 'preserve')
            pos = stop


def set_text(paragraph, text):
    assert not paragraph._p.xpath('.//m:oMath|.//w:drawing')
    if paragraph.runs:
        for run in paragraph.runs:
            run.text = ''
        paragraph.runs[0].text = text
    else:
        paragraph.add_run(text)


def add(anchor, text, style='Normal'):
    new = anchor.insert_paragraph_before(text, style=style)
    if style == 'Normal' and p[100]._p.pPr is not None:
        if new._p.pPr is not None:
            new._p.remove(new._p.pPr)
        new._p.insert(0, deepcopy(p[100]._p.pPr))
    return new


def add_table(anchor, caption, columns, data):
    cap = add(anchor, caption, 'Caption')
    cap.paragraph_format.keep_with_next = True
    table = doc.add_table(rows=0, cols=len(columns))
    table.style = 'Table Grid'
    for index, values in enumerate((columns, *data)):
        row = table.add_row()
        for cell, value in zip(row.cells, values):
            para = cell.paragraphs[0]
            para.paragraph_format.space_before = Pt(2)
            para.paragraph_format.space_after = Pt(2)
            para.paragraph_format.keep_with_next = index < len(data)
            run = para.add_run(value)
            run.font.size = Pt(9)
            run.bold = index == 0
    anchor._p.addprevious(table._tbl)


for paragraph in doc.element.body.xpath('.//w:p'):
    replace(paragraph, '大视差', '大重投影位移')
set_text(p[0], '复杂遮挡下的可见性感知多视图深度估计')
replace(p[41]._p, '对于被遮挡样本，不确定性损失停止误差项向成对深度预测传播梯度，保留对不确定性分支的优化。由此，网络能够减少无有效对应样本对成对深度回归的干扰，并学习这些样本的低可靠性。',
        '对于遮挡像素，不确定性损失截断成对深度误差项的直接梯度。成对匹配概率的熵参与不确定性估计，因此该损失仍可经不确定性分支更新匹配打分头和共享匹配网络。')
replace(p[48]._p, '并使被遮挡样本的不确定性损失只更新不确定性分支', '并截断遮挡样本成对深度误差的直接梯度')
replace(p[84]._p, '本文“大重投影位移”采用上述重投影位移定义，包含水平和垂直分量，并非校正双目中的纯水平视差。', '本文将这一区域称为大重投影位移区域，位移包含水平和垂直分量。')
replace(p[3]._p, '从7.281降至5.493', '从7.281 mm降至5.493 mm')

add(p[54], '固定配置以阶段名义间距生成候选，即从前级预测减去固定半宽的位置出发，按给定步长取候选。候选数为偶数时，下端距预测中心比上端多一个步长。自适应配置在裁剪后的两端点之间等距取样，名义间距用于确定半宽约束，实际间距由裁剪后窗宽和候选数共同决定。')
add(p[57], '范围调节只作用于第二、三级。候选数量保持一致，搜索窗改变后，候选间距由搜索窗宽度决定。')
add(p[60], '三个阶段均可使用候选相关融合。推理对每个深度候选的各源视图分数进行归一化，分数由负对数不确定性与有界候选残差相加得到。辅助可见性概率用于训练监督和诊断。')
add(p[76], '式（20）的停止梯度算子只作用于遮挡像素的成对深度误差。不确定性由匹配概率的熵得到，匹配打分头及共享匹配网络仍可沿这条路径接收梯度。候选可见性损失启用时，若真值落在候选区间外，最近候选规则会选取相应端点。')

set_text(p[100], '表1—4沿用八组配置的展示名称。A、B、C用于指称本文讨论的逐源遮挡监督、自适应深度搜索和深度假设感知源视图融合。各行数值在相同真值有效掩码与困难区域掩码上比较。展示名称与检查点实际开关分别记录，例如Base、Base+B和完整配置对应的历史DTU结果标识为range（010）、vis（000）和hyp（001）。本节以配置间差异描述历史数值，实际模块的作用由第5.4节的受控实验检验。')
set_text(p[104], '以论文表中的Base+B配置为例，五视图训练时整体Abs从7.281 mm降至5.596 mm，大重投影位移区域从21.469 mm降至14.730 mm，交集从58.326 mm降至46.688 mm。三视图训练时，大重投影位移区域的Abs从22.940 mm降至17.089 mm。这些数值描述所列配置之间的差异。搜索窗的覆盖与最终误差由第5.4节在固定前级输出和固定检查点的条件下进一步检验。')
set_text(p[106], '五视图训练时，表中Base+A相对Base将任一源遮挡区域的Abs从35.936 mm降至34.756 mm，将边界与遮挡交集的Abs从38.742 mm降至37.102 mm。三视图训练时，完整配置在大重投影位移与遮挡交集的Abs为47.328 mm，表中Base+B+C配置为50.825 mm。A的梯度机制见第5.4.1节。')
set_text(p[107], '三视图训练时，表中Base+C相对Base的整体Abs从7.634 mm降至7.612 mm，交集Abs从62.056 mm降至59.772 mm。完整配置与Base+A+B相比，交集Abs从51.307 mm降至47.328 mm。第5.4.3节进一步固定检查点与候选，测量候选残差对实际权重和最终深度的影响。')
replace(p[108]._p, '因此，现有消融更有力地支持三个环节联合作用的区域改善，而各模块的收益大小仍受到当前组合和训练设置的影响。', '这表明配置之间的差异随评价区域和训练设置变化。')
for table in tables[39:41]:
    for cell in table.rows[0].cells:
        for paragraph in cell.paragraphs:
            replace(paragraph._p, 'Abs/mm', 'Abs（归一化单位）')
add(p[94], 'BlendedMVS以场景相机有效深度上界的中位数为尺度基准。参考深度与相机平移按同一场景倍率缩放，遮挡判断的绝对深度容差也在归一化尺度下定义。表8和表9以归一化深度单位报告Abs。')

anchor = p[157]
add(anchor, '5.4 机制验证', 'Heading 2')
add(anchor, '受控实验沿用已训练检查点与五视图推理。B、C的测试各覆盖22个DTU扫描，固定取每个扫描第13号参考视图及光照3。困难交集由图内大重投影位移与至少一源遮挡共同定义。A在scan4的两个固定参考视图上核验，每个参考视图使用四个源视图。')
add(anchor, '5.4.1 遮挡监督的梯度路径', 'Heading 3')
add(anchor, '用含遮挡监督的oa_hyp（101）分别对可见与遮挡像素的不确定性损失反向传播，比较式（20）的截断规则与去掉截断的对照。16个遮挡组的成对深度输出直接梯度均为零，去掉截断后其L2范数为0.0048–0.0252。启用截断时，匹配打分头、不确定性头和共享匹配正则化网络在16组中均收到梯度，范数中位数分别为2.28、3.24和6.10。可见组的直接梯度在两种规则下相同。两种规则的损失值一致，说明截断只改变反传路径。')
add(anchor, '第三级的遮挡和可见记录中，真值落在区间外的比例分别为19.86%和6.85%。当候选可见性损失启用时，区间外样本将监督最近的端点候选。该比例来自当前小样本的几何标签与候选诊断。')
add(anchor, '5.4.2 搜索窗覆盖与最终深度', 'Heading 3')
add(anchor, '对vis（000）检查点固定前级输出，在相同物理深度边界和候选数量下比较固定窗、无界κσ窗与有界κσ窗。进一步在18个验证扫描上将固定窗校准到有界窗的全图平均宽度，再将窗宽冻结用于独立测试。第三级离线对照中，三视图和五视图训练设置的有界窗全图平均窗宽分别减少0.18和0.20 mm，困难交集的覆盖率分别提高0.90和0.94个百分点。困难交集的搜索窗分别增宽1.99和1.79 mm，最近候选MAE由32.67降至31.94 mm、由35.89降至35.23 mm。')
add(anchor, '随后固定同一检查点和前两级输出，只替换第三级候选并完成深度回归。表10报告困难交集上的配对结果。仅有界窗覆盖的像素分别有168和180个，其中分别有23和29个由大于8 mm降至不超过8 mm，也分别有4和3个反向跨过阈值。在双方均覆盖的像素上，MAE分别由4.44升至4.51 mm、由4.05升至4.12 mm。扩大困难像素的候选覆盖可以纠正部分粗差，交集整体MAE和P95在本对照中略升。')
add_table(anchor, '表10 仅替换第三级搜索窗对困难交集的影响（固定窗 → 有界窗）',
          ('训练视图数', 'GT覆盖率', 'MAE（mm）', 'P95（mm）', '粗差率'),
          [('3', '55.85% → 56.79%', '39.46 → 39.56', '159.35 → 160.27', '54.05% → 53.93%'),
           ('5', '57.98% → 58.99%', '42.25 → 42.28', '214.99 → 215.35', '50.32% → 50.28%')])
add(anchor, '5.4.3 候选残差与实际融合权重', 'Heading 3')
add(anchor, '对同一个hyp（001）检查点，只在第三级保留或置零候选相关残差，核对前两级预测、第三级候选及覆盖掩码一致。在困难交集中，选取真值已覆盖且同时存在可见源和遮挡源的像素，于最近真值候选处汇总已标记遮挡源的归一化权重。开启残差后，两种训练设置的遮挡源权重平均相对下降约23%–24%，均有19/21个有效扫描下降。表11同时给出最终误差，三视图和五视图训练的逐扫描配对MAE变化的95%重采样区间均跨过零。')
add_table(anchor, '表11 第三级候选残差的受控对照（关闭 → 开启）',
          ('训练视图数', '像素数', '遮挡源权重和', 'MAE（mm）', 'P95（mm）', '粗差率'),
          [('3', '10 083', '0.317 → 0.242', '4.668 → 4.666', '14.97 → 14.99', '19.88% → 19.84%'),
           ('5', '10 924', '0.340 → 0.261', '3.784 → 3.774', '12.98 → 12.67', '12.55% → 12.25%')])
add(anchor, '这些对照分别验证了训练梯度、候选覆盖和实际融合权重的变化。第三级单独干预下，困难区域的最终误差变化较小，候选覆盖及遮挡源权重的改善程度不能直接作为最终深度误差降幅。')

replace(p[159]._p, '消融结果进一步显示，自适应范围带来了较大的大重投影位移区域收益，遮挡监督和候选相关融合在已有组合上能够继续改善部分目标区域。', '受控实验表明，遮挡监督截断了遮挡像素的直接深度误差梯度，有界搜索提高了困难交集的真值覆盖，候选相关残差降低了已标记遮挡源的融合权重。单独调整第三级搜索窗或残差后，该交集的最终误差变化较小。')

doc.save(output)
assert sha256(source.read_bytes()).hexdigest() == original_hash
reopened = Document(output)
assert len(reopened.tables) == len(tables) + 2
assert all([[c.text for c in row.cells if c.text] for row in reopened.tables[i].rows[1:]] ==
           [[c.text for c in row.cells if c.text] for row in Document(source).tables[i].rows[1:]]
           for i in range(26,30))
assert len(reopened.element.body.xpath('.//m:oMath')) == len(Document(source).element.body.xpath('.//m:oMath'))
print(output)
