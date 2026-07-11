"""Render the compiled support-agent graph for visual inspection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.build import build_graph

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "graph"

if __name__ == "__main__":
    graph = build_graph()
    graph_repr = graph.get_graph()

    print("=== ASCII rendering ===")
    try:
        graph_repr.print_ascii()
    except ImportError as e:
        print(f"ASCII rendering skipped ({e}) - continuing to Mermaid export.")

    mermaid_text = graph_repr.draw_mermaid()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mermaid_path = OUTPUT_DIR / "router_graph.mmd"
    mermaid_path.write_text(mermaid_text)
    print(f"\nMermaid diagram source written to: {mermaid_path}")
    print("(paste its contents into https://mermaid.live to view, or render via a Mermaid-aware Markdown viewer)")

    try:
        png_bytes = graph_repr.draw_mermaid_png()
        png_path = OUTPUT_DIR / "router_graph.png"
        png_path.write_bytes(png_bytes)
        print(f"PNG rendering written to: {png_path}")
    except Exception as e:
        print(f"PNG rendering failed ({type(e).__name__}: {e}) - this needs network access to Mermaid's render API, the .mmd file above is a reliable fallback.")
