"""Side-by-side AA/CG visualisation of ligands in examples.

Generates an interactive HTML page per ligand showing the AA and CG
topology side by side using py3Dmol.

Usage::

    python scripts/visualize.py              # all ligands
    python scripts/visualize.py ANP CLA      # specific ligands
    python scripts/visualize.py --open ANP   # generate and open in browser
"""

import logging
import sys
import webbrowser
from pathlib import Path

import smartini
from smartini.config import CFG

logger = logging.getLogger(__name__)
smartini.setup_logging(level=logging.INFO)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
EXAMPLES_DIR = Path("examples").resolve()
OUTPUT_DIR = Path("analysis/views").resolve()

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 16px; background: #1a1a2e; color: #eee; }}
  h1 {{ text-align: center; }}
  .grid {{ display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }}
  .panel {{ flex: 1 1 450px; max-width: 600px; }}
  .panel h2 {{ text-align: center; margin: 4px 0; }}
  .viewer {{ width: 100%; height: 450px; position: relative; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="grid">
  <div class="panel">
    <h2 style="color:#88ccff">AA  (atomistic)</h2>
    <div class="viewer" id="aa_view"></div>
  </div>
  <div class="panel">
    <h2 style="color:#ffaa44">CG  (coarse-grained)</h2>
    <div class="viewer" id="cg_view"></div>
  </div>
</div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let $ = $3Dmol.noConflict();

  function mkViewer(id, pdb, style) {{
    let v = $(id, {{ backgroundColor: "0x1a1a2e" }});
    v.addModel(pdb, "pdb");
    style(v);
    v.zoomTo();
    v.render();
    return v;
  }}

  // Sync rotation & zoom between the two viewers
  function linkViews(a, c) {{
    let dragging = false;
    a.setCallback("ondragstart", () => {{ dragging = true; }});
    a.setCallback("ondrag", (rot) => {{ c.setRotation(rot); }});
    a.setCallback("ondragend", () => {{ dragging = false; }});
    c.setCallback("ondragstart", () => {{ dragging = true; }});
    c.setCallback("ondrag", (rot) => {{ a.setRotation(rot); }});
    c.setCallback("ondragend", () => {{ dragging = false; }});
    let zooming = false;
    a.setCallback("onzoomstart", () => {{ zooming = true; }});
    a.setCallback("onzoom", (z) => {{ if (zooming) c.zoom(z); }});
    a.setCallback("onzoomend", () => {{ zooming = false; }});
    c.setCallback("onzoomstart", () => {{ zooming = true; }});
    c.setCallback("onzoom", (z) => {{ if (zooming) a.zoom(z); }});
    c.setCallback("onzoomend", () => {{ zooming = false; }});
  }}

  mkViewer("aa_view", `{aa_pdb}`, v => v.setStyle({{stick: {{colorscheme: "cyanCarbon"}}}}));
  mkViewer("cg_view", `{cg_pdb}`, v => {{
    v.setStyle({{sphere: {{radius: 0.3, colorscheme: "orangeCarbon"}}}});
    v.addModel(`{cg_pdb}`, "pdb");
    v.setStyle({{model:1}}, {{stick: {{radius: 0.08, colorscheme: "orangeCarbon"}}}});
  }});
  linkViews($("#aa_view"), $("#cg_view"));
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def generate_view(ligand_name: str, out_dir: Path | None = None) -> Path | None:
    """Generate a side-by-side AA/CG HTML view for one ligand.

    Returns the path to the generated HTML file, or None if data is missing.
    """
    aa_pdb = EXAMPLES_DIR / ligand_name / "aa_md" / "topology.pdb"
    cg_pdb = EXAMPLES_DIR / ligand_name / "cg_md" / CFG.cg_runname / "topology.pdb"

    if not aa_pdb.exists():
        logger.warning("[%s] AA topology not found: %s", ligand_name, aa_pdb)
        return None
    if not cg_pdb.exists():
        logger.warning("[%s] CG topology not found: %s", ligand_name, cg_pdb)
        return None

    if out_dir is None:
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    html = _HTML_TEMPLATE.format(
        title=f"{ligand_name} &mdash; AA vs CG",
        aa_pdb=aa_pdb.read_text(),
        cg_pdb=cg_pdb.read_text(),
    )

    out_path = out_dir / f"{ligand_name}.html"
    out_path.write_text(html)
    logger.info("[%s] View saved → %s", ligand_name, out_path)
    return out_path


def generate_index(ligand_names: list[str], out_dir: Path):
    """Generate an index.html that iframe-embeds all ligand views."""
    first = sorted(ligand_names)[0]
    items = "\n".join(
        f'    <li><a href="{n}.html" target="view">{n}</a></li>'
        for n in sorted(ligand_names)
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SMartini Ligand Views</title>
<style>
  body {{ font-family: sans-serif; margin: 0; background: #1a1a2e; color: #eee; }}
  .layout {{ display: flex; height: 100vh; }}
  nav {{ width: 200px; padding: 12px; border-right: 1px solid #333; overflow-y: auto; }}
  nav h2 {{ font-size: 14px; color: #888; }}
  nav ul {{ list-style: none; padding: 0; }}
  nav li {{ margin: 4px 0; }}
  nav a {{ color: #88ccff; text-decoration: none; }}
  nav a:hover {{ color: #fff; }}
  iframe {{ flex: 1; border: none; }}
</style></head>
<body>
<div class="layout">
  <nav>
    <h2>Ligands</h2>
    <ul>{items}</ul>
  </nav>
  <iframe name="view" src="{first}.html"></iframe>
</div>
</body></html>"""
    idx_path = out_dir / "index.html"
    idx_path.write_text(html)
    logger.info("Index saved → %s", idx_path)
    return idx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _discover_ligands() -> list[str]:
    """Return sorted ligand names that have both AA and CG topology PDBs."""
    ligands = []
    for d in sorted(EXAMPLES_DIR.iterdir()):
        if not d.is_dir():
            continue
        aa = d / "aa_md" / "topology.pdb"
        cg = d / "cg_md" / CFG.cg_runname / "topology.pdb"
        if aa.exists() and cg.exists():
            ligands.append(d.name)
    return ligands


if __name__ == "__main__":
    open_browser = "--open" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--open"]
    names = args if args else _discover_ligands()

    if not names:
        logger.error("No ligands with AA+CG topology found in examples/")
        sys.exit(1)

    logger.info("Generating views for %d ligand(s): %s", len(names), ", ".join(names))
    for name in sorted(names):
        generate_view(name)

    idx = generate_index(sorted(names), OUTPUT_DIR)
    if open_browser:
        webbrowser.open(idx.as_uri())
        logger.info("Opened %s", idx.as_uri())
    logger.info("Done.")
