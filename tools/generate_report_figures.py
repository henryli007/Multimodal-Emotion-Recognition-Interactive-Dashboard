from pathlib import Path
from xml.sax.saxutils import escape
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
GRAPH = ROOT / "workspace" / "lightrag" / "graph_chunk_entity_relation.graphml"

STYLE = '''
<style>
svg{background:#fff}.title{font:700 22px Arial,"Noto Sans CJK SC",sans-serif;fill:#111827}.sub{font:400 13px Arial,"Noto Sans CJK SC",sans-serif;fill:#4b5563}.h{font:700 14px Arial,"Noto Sans CJK SC",sans-serif;fill:#111827}.t{font:400 13px Arial,"Noto Sans CJK SC",sans-serif;fill:#1f2937}.s{font:400 12px Arial,"Noto Sans CJK SC",sans-serif;fill:#4b5563}.tiny{font:400 11px Arial,"Noto Sans CJK SC",sans-serif;fill:#6b7280}.box{fill:#fff;stroke:#111827;stroke-width:1.2}.soft{fill:#f8fafc;stroke:#cbd5e1;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.edge{stroke:#374151;stroke-width:1.2;fill:none}.dash{stroke:#6b7280;stroke-width:1.1;stroke-dasharray:4 4;fill:none}.accent{fill:#2563eb}.audio{fill:#d97706}.textc{fill:#2563eb}.fused{fill:#059669}.evidence{fill:#7c3aed}
</style>'''

def svg(w,h,body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{STYLE}{body}</svg>'

def label(x,y,text,cls='t',anchor='start'):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(str(text))}</text>'

def rect(x,y,w,h,cls='box'):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}"/>'

def case_trace():
    b=[label(48,38,'单轮会话的系统运行产物', 'title'),label(48,61,'同一句输入在语义、韵律、融合、图谱检索与数字人回复中的可观测结果', 'sub')]
    for y in (112,192,272,352): b.append(f'<line class="grid" x1="48" y1="{y}" x2="1152" y2="{y}"/>')
    cols=[48,232,438,646,872]
    heads=['用户输入','模态识别','融合状态','图谱证据','数字人疏导']
    widths=[160,182,184,202,280]
    for x,h,w in zip(cols,heads,widths):
        b += [rect(x,82,w,30,'soft'),label(x+w/2,102,h,'h','middle')]
    b += [rect(48,124,160,196),label(64,152,'“我没事，', 'h'),label(64,174,'只是有点撑不住了。”','h'),label(64,226,'语音：明亮、语速快','s'),label(64,248,'文本：负性语义','s')]
    b += [rect(232,124,182,82),label(248,149,'文本语义', 'h'),label(248,174,'悲伤 62%  紧张 21%','t'),rect(232,218,182,82),label(248,243,'语音韵律','h'),label(248,268,'高兴 55%  平静 24%','t')]
    b += [rect(438,124,184,176),label(454,149,'加权融合', 'h'),label(454,176,'0.55 × 文本', 't'),label(454,198,'0.45 × 语音', 't'),label(454,236,'最终状态', 'h'),label(454,262,'悲伤 39%  高兴 25%','t'),label(454,284,'紧张 18%','t')]
    b += [rect(646,124,202,176),label(662,149,'命中节点', 'h'),label(662,176,'压力', 't'),label(662,198,'情绪压抑', 't'),label(662,220,'深呼吸', 't'),label(662,242,'设定小目标', 't'),label(662,276,'检索：LightRAG hybrid','s')]
    b += [rect(872,124,280,176),label(888,149,'回复条件', 'h'),label(888,176,'融合情绪 + 原话 + 图谱证据','t'),label(888,214,'“听起来你在努力维持轻松，', 't'),label(888,236,'但其实已经很累了……”','t'),label(888,274,'TTS → 数字人视频','s')]
    for x1,x2 in [(208,232),(414,438),(622,646),(848,872)]: b.append(f'<path class="edge" d="M{x1},212 H{x2}"/>')
    b.append(label(48,375,'图 1  语气与语义冲突时，系统不会把单一模态当作真值，而是保留两路证据后形成可解释的融合状态。','s'))
    return svg(1200,400,''.join(b))

def fusion_panel():
    b=[label(48,38,'多模态情绪融合示例', 'title'),label(48,61,'文本与语音并列展示；右侧为送入图谱检索与回复生成的最终分布', 'sub')]
    labels=['悲伤','高兴','紧张','平静']; text=[62,7,21,10]; audio=[8,55,13,24]; fused=[39,25,18,18]
    for x,h,c in [(210,'文本语义','#2563eb'),(500,'语音韵律','#d97706'),(790,'融合输出','#059669')]:
        b += [label(x,102,h,'h'),f'<line x1="{x}" y1="114" x2="{x+190}" y2="114" stroke="{c}" stroke-width="3"/>']
    for i,l in enumerate(labels):
        y=148+i*56; b += [label(72,y+14,l,'h')]
        for x,v,c in [(210,text[i],'#2563eb'),(500,audio[i],'#d97706'),(790,fused[i],'#059669')]:
            b += [f'<rect x="{x}" y="{y}" width="190" height="18" fill="#f3f4f6"/>',f'<rect x="{x}" y="{y}" width="{v*1.9}" height="18" fill="{c}"/>',label(x+202,y+14,f'{v}%','t')]
    b += [f'<line class="dash" x1="705" y1="88" x2="705" y2="370"/>',rect(48,392,1060,58,'soft'),label(64,415,'解释：用户可以“用高兴语气说悲伤的话”。系统因此保留语音证据，但让融合结果同时受语义约束。','t'),label(64,438,'融合输出才进入实时情绪感知面板、图谱检索提示和数字人回复生成。','t')]
    return svg(1160,470,''.join(b))

def graph_evidence():
    g=nx.read_graphml(GRAPH); center='焦虑'; ns=sorted(g.neighbors(center), key=lambda n:g.degree(n), reverse=True)[:10]
    xs=[210,390,570,750,930]; ys=[150,225]
    b=[label(48,38,'图谱证据子图', 'title'),label(48,61,'从真实知识图谱裁剪的局部证据；布局采用固定网格，便于论文排版与逐项标注', 'sub'),rect(48,92,1064,260,'soft')]
    b += [f'<circle cx="120" cy="222" r="34" fill="#111827"/>',label(120,227,center,'h','middle')]
    for i,n in enumerate(ns):
        x=xs[i%5]; y=ys[i//5]; typ=g.nodes[n].get('entity_type','UNKNOWN'); color={'疏导举措':'#059669','诱发因素':'#d97706'}.get(typ,'#2563eb')
        b += [f'<line x1="154" y1="222" x2="{x-70}" y2="{y}" stroke="#9ca3af" stroke-width="1.2"/>',f'<rect x="{x-70}" y="{y-20}" width="140" height="40" fill="#fff" stroke="{color}" stroke-width="1.5"/>',label(x,y+5,n,'t','middle')]
    b += [label(48,385,'节点类型', 'h'),f'<rect x="124" y="372" width="12" height="12" fill="#2563eb"/>',label(144,383,'情绪/症状','s'),f'<rect x="244" y="372" width="12" height="12" fill="#d97706"/>',label(264,383,'诱发因素','s'),f'<rect x="364" y="372" width="12" height="12" fill="#059669"/>',label(384,383,'疏导举措','s')]
    return svg(1160,420,''.join(b))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    figures={'figure_1_case_trace.svg':case_trace(),'figure_2_multimodal_fusion.svg':fusion_panel(),'figure_3_graph_evidence.svg':graph_evidence()}
    for name,content in figures.items():
        (OUT/name).write_text(content,encoding='utf-8'); print(OUT/name)

if __name__=='__main__': main()
