import os
from pathlib import Path

import networkx as nx
from pyvis.network import Network
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
graphml_path = PROJECT_ROOT / "workspace" / "lightrag" / "graph_chunk_entity_relation.graphml"


def rewrite_asset_paths(html: str) -> str:
    # The exported file lives under workspace/, so local pyvis assets must go up
    # one level to reach the repo-root lib/ directory when served over HTTP.
    return (
        html.replace('src="lib/', 'src="../lib/')
        .replace("src='lib/", "src='../lib/")
        .replace('href="lib/', 'href="../lib/')
        .replace("href='lib/", "href='../lib/")
    )

def visualize():
    if not os.path.exists(graphml_path):
        print(f"【错误】找不到图谱文件: {graphml_path}")
        print("请确定 LightRAG 知识库至少完成了一次构建任务并生成了 graphml 文件。")
        return

    print(f"正在加载知识图谱: {graphml_path} ...")
    G = nx.read_graphml(graphml_path)
    
    print(f"加载完毕！该图谱当前包含 {G.number_of_nodes()} 个节点， {G.number_of_edges()} 条关系边。")
    print("正在生成可交互式网页可视化界面...")
    
    # 建立 Pyvis 网络
    # 配置深色背景与物理引擎以实现节点之间的自动排布
    net = Network(
        height='100vh', 
        width='100%', 
        bgcolor='#1E1E1E', 
        font_color='white',
        select_menu=True,   # 提供选择菜单（选择某个节点高亮）
        filter_menu=True    # 提供过滤菜单
    )
    net.barnes_hut(gravity=-8000) # 物理引力模型，防止节点重叠太紧密
    
    # 遍历节点添加悬浮提示（title）和标签（label）
    for node, data in G.nodes(data=True):
        # 悬浮显示完整的属性信息
        hover_info = "<br>".join([f"{k}: {v}" for k, v in data.items()])
        data['title'] = hover_info
        
        # 节点上显示的文本，如果有 entity_name 则优先显示，否则显示 id
        if 'entity_name' in data:
            data['label'] = str(data['entity_name'])
        elif 'id' in data:
            data['label'] = str(data['id'])
        else:
            data['label'] = str(node)

    # 从 NetworkX 将图汇入 pyvis
    net.from_nx(G)
    
    output_file = PROJECT_ROOT / "workspace" / "knowledge_graph_visualization.html"
    # 生成 HTML 并保存（通过生成字符串后手动以 utf-8 编码写入文件，规避 pyvis 的 GBK 默认写入错误）
    html_content = rewrite_asset_paths(net.generate_html())
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    full_path = os.path.abspath(output_file)
    print(f"\n✅ 可视化页面生成成功！\n文件储存在: {full_path}")
    print("正在为您尝试通过默认浏览器自动打开...")
    
    webbrowser.open(f"file://{full_path}")

if __name__ == "__main__":
    visualize()
