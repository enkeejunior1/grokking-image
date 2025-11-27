import colorsys
import pydot

def generate_rainbow_hex_colors(N):
    """
    N개의 레이어에 고르게 분포된 무지개색 HEX 코드 리스트를 생성합니다.
    """
    hex_colors = []
    MAX_HUE = 0.85 
    
    for i in range(N):
        if N == 1:
            hue_fraction = 0.0
        else:
            hue_fraction = i / (N - 1) * MAX_HUE
        
        rgb_float = colorsys.hsv_to_rgb(hue_fraction, 1.0, 1.0)
        
        r = int(rgb_float[0] * 255)
        g = int(rgb_float[1] * 255)
        b = int(rgb_float[2] * 255)
        
        hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
        hex_colors.append(hex_code)
        
    return hex_colors


def draw_network(n_layers, n_heads, deactivated_heads, results_dir, zero_out=False, stats=None):
    if stats is not None:
        label = f"Accuracy: {stats[0]:.3f}, Avg. Confidence: {stats[1]:.3f}"
    else:
        label = "Transformer Structure"
    graph_attn = pydot.Dot("transformer_flow", graph_type="digraph", rankdir="LR", splines="line", label=label, labelloc="t", fontsize="20") 
    layer_colors = generate_rainbow_hex_colors(n_layers)
    rad = "0.2"
    for l in range(n_layers):
        curr_cluster = pydot.Cluster(f"cluster_L{l+1}", label=f"Layer {l+1}", color="lightgrey", style="filled", fillcolor="#F9F9F9")
        
        for h in range(n_heads):
            if (l, h) in deactivated_heads:
                head_node = pydot.Node(f"L{l+1}_H{h+1}", label="", width=rad, height=rad, fixed_size="true", shape="circle", style="filled", fillcolor="grey", penwidth="0")
            else:
                head_node = pydot.Node(f"L{l+1}_H{h+1}", label="", width=rad, height=rad, fixed_size="true", shape="circle", style="filled", fillcolor=layer_colors[l], penwidth="0")
            curr_cluster.add_node(head_node)
        
        graph_attn.add_subgraph(curr_cluster)
        
        if l > 0:
            for h_prev in range(n_heads):
                for h_curr in range(n_heads):
                    if (l-1, h_prev) not in deactivated_heads and (l, h_curr) not in deactivated_heads:
                        edge = pydot.Edge(f"L{l}_H{h_prev+1}", f"L{l+1}_H{h_curr+1}", style="solid", color="black", arrowhead="normal", arrowsize="0.2")
                    else:
                        if zero_out:
                            edge = pydot.Edge(f"L{l}_H{h_prev+1}", f"L{l+1}_H{h_curr+1}", style="solid", color="#00000000", arrowhead="normal", arrowsize="0.2")
                        else:
                            edge = pydot.Edge(f"L{l}_H{h_prev+1}", f"L{l+1}_H{h_curr+1}", style="solid", color="#80808033", arrowhead="normal", arrowsize="0.2")
                    graph_attn.add_edge(edge)
                        
    
    graph_attn.write_png(f"transformer_structure.png")


if __name__ == "__main__":
    n_layers = 6
    n_heads = 8
    deactivated_heads = set()
    from random import random
    for l in range(n_layers):
        for h in range(n_heads):
            if random() < 0.3:
                deactivated_heads.add((l, h))
    results_dir = "."
    stats = (0.85, 0.92)
    
    draw_network(n_layers, n_heads, deactivated_heads, results_dir, zero_out=True, stats=stats)