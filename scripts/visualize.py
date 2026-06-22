"""Side-by-side AA/CG visualisation of ligands in examples.

Generates interactive HTML pages using py3Dmol.
CG beads are rendered as semi-transparent spheres (SASA-rule radii:
T* → 0.17, S* → 0.205, else → 0.235 nm) plus sticks from CONECT records.

Usage::

    python scripts/visualize.py              # all ligands
    python scripts/visualize.py ANP CLA      # specific ligands
    python scripts/visualize.py --open ANP   # generate and open in browser
"""

import json
import logging
import re
import sys
import webbrowser
from collections import defaultdict
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


# ---------------------------------------------------------------------------
# ITP helpers  (same as analysis.py — self-contained so script runs standalone)
# ---------------------------------------------------------------------------

def _parse_itp(itp_path: Path) -> tuple[list[list[int]], list[str]]:
    """Parse ``LIGAND.itp`` → (mapping, bead_types)."""
    text = itp_path.read_text()
    m = re.search(r"; Mapping:\s*(\[\[.*?\]\])", text)
    if not m:
        raise ValueError(f"No '; Mapping:' found in {itp_path}")
    mapping = [list(map(int, g.strip("[] ").split(",")))
               for g in re.findall(r"\[([^\]]+)\]", m.group(1))]
    in_atoms = False
    bead_types = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_atoms = (s == "[ atoms ]" or s == "[atoms]")
            continue
        if in_atoms and s and not s.startswith(";") and not s.startswith("#"):
            parts = s.split()
            if len(parts) >= 2:
                bead_types.append(parts[1])
    return mapping, bead_types


def _bead_radius(bead_type: str) -> float:
    """Martini bead radius (nm): T* → 0.17, S* → 0.205, else → 0.235."""
    if bead_type.startswith("T"):
        return 0.17
    if bead_type.startswith("S"):
        return 0.205
    return 0.235

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
(function() {{
  var Viewer = $3Dmol.createViewer;

  // --- AA ---
  var aa = Viewer("aa_view", {{ backgroundColor: "0x1a1a2e" }});
  aa.addModel(`{aa_pdb}`, "pdb");
  aa.setStyle({{}}, {{ stick: {{ colorscheme: "cyanCarbon" }} }});
  aa.zoomTo();
  aa.render();

  // --- CG ---
  var cg = Viewer("cg_view", {{ backgroundColor: "0x1a1a2e" }});
  // sticks from CONECT records
  cg.addModel(`{cg_pdb}`, "pdb");
  cg.setStyle({{}}, {{ stick: {{ radius: 0.08, colorscheme: "orangeCarbon" }} }});
  // semi-transparent spheres — one model per bead type (array-based selector)
  var beadRadii = {bead_radii_json};
  var beadGroups = {bead_groups_json};
  for (var bt in beadGroups) {{
    var r = beadRadii[bt];
    var atomSel = beadGroups[bt].map(function(i) {{ return i + 1; }}); // 1‑based
    cg.addModel(`{cg_pdb}`, "pdb");
    cg.setStyle({{ atom: atomSel }}, {{ sphere: {{ radius: r, opacity: 0.55, colorscheme: "orangeCarbon" }} }});
  }}
  cg.zoomTo();
  cg.render();

  // --- Sync rotation ---
  var dragging = false;
  aa.setCallback("ondragstart", function() {{ dragging = true; }});
  aa.setCallback("ondrag", function(rot) {{ cg.setRotation(rot); if (dragging) cg.render(); }});
  aa.setCallback("ondragend", function() {{ dragging = false; }});
  cg.setCallback("ondragstart", function() {{ dragging = true; }});
  cg.setCallback("ondrag", function(rot) {{ aa.setRotation(rot); if (dragging) aa.render(); }});
  cg.setCallback("ondragend", function() {{ dragging = false; }});
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_view(ligand_name: str, out_dir: Path | None = None) -> Path | None:
    """Generate a side-by-side AA/CG HTML view for one ligand.

    Returns the path to the generated HTML file, or None if data is missing.
    """
    aa_pdb = EXAMPLES_DIR / ligand_name / "aa_md" / "topology.pdb"
    cg_pdb = EXAMPLES_DIR / ligand_name / f"{ligand_name}.pdb"       # has CONECT
    cg_top = EXAMPLES_DIR / ligand_name / "cg_md" / CFG.cg_runname / "topology.pdb"
    itp_path = EXAMPLES_DIR / ligand_name / f"{ligand_name}.itp"

    if not aa_pdb.exists():
        logger.warning("[%s] AA topology not found: %s", ligand_name, aa_pdb)
        return None
    if not cg_top.exists():
        logger.warning("[%s] CG data not found: %s", ligand_name, cg_top)
        return None
    if not cg_pdb.exists():
        logger.warning("[%s] CG PDB (with CONECT) not found: %s", ligand_name, cg_pdb)
        return None

    if out_dir is None:
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build bead-type → radius and bead-type → atom-index lists for JS
    _, bead_types = _parse_itp(itp_path)
    bead_radii = {bt: _bead_radius(bt) for bt in set(bead_types)}
    bead_groups = defaultdict(list)
    for i, bt in enumerate(bead_types):
        bead_groups[bt].append(i)

    html = _HTML_TEMPLATE.format(
        title=f"{ligand_name} &mdash; AA vs CG",
        aa_pdb=aa_pdb.read_text(),
        cg_pdb=cg_pdb.read_text(),
        bead_radii_json=json.dumps(bead_radii),
        bead_groups_json=json.dumps(dict(bead_groups)),
    )

    out_path = out_dir / f"{ligand_name}.html"
    out_path.write_text(html)
    logger.info("[%s] HTML view saved → %s", ligand_name, out_path)
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

    logger.info("Generating HTML views for %d ligand(s): %s", len(names), ", ".join(names))
    for name in sorted(names):
        generate_view(name)

    idx = generate_index(sorted(names), OUTPUT_DIR)
    if open_browser:
        webbrowser.open(idx.as_uri())
        logger.info("Opened %s", idx.as_uri())
    logger.info("Done.")
