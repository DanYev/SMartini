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

# --- Visual style (tweak for publication) ---
BG_COLOR           = "0xffffff"   # 3Dmol hex: white
CG_STICK_COLOR     = "#cc6622"    # orange-brown sticks
CG_SPHERE_COLOR    = "#ee8833"    # orange spheres
CG_SPHERE_OPACITY  = 0.50         # 0–1
CG_SPHERE_SCALE    = 10           # nm → Å display multiplier
CG_X_OFFSET        = 40.0         # Å — shift CG to the right of AA


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


def _offset_pdb(pdb_text: str, dx: float) -> str:
    """Shift all ATOM/HETATM X coordinates by *dx* Å; return new PDB string."""
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) + dx
            lines.append(f"{line[:30]}{x:8.3f}{line[38:]}")
        else:
            lines.append(line)
    return "\n".join(lines)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 8px; background: #fff; color: #222; }}
  h1 {{ text-align: center; margin: 0 0 4px 0; font-size: 16px; }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: 8px 20px; justify-content: center;
    align-items: center; padding: 6px 12px; font-size: 13px;
    background: #f5f5f5; border-bottom: 1px solid #ccc; margin-bottom: 4px;
  }}
  .controls label {{ display: flex; align-items: center; gap: 4px; white-space: nowrap; }}
  .controls input[type="color"] {{ width: 28px; height: 22px; border: 1px solid #999; padding: 0; cursor: pointer; }}
  .controls input[type="range"] {{ width: 90px; }}
  .controls .val {{ display: inline-block; width: 32px; text-align: right; }}
  .viewer {{ width: 100%; height: 520px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="controls">
  <label>CG sticks <input type="color" id="cg_stick" value="{cg_stick_color}"></label>
  <label>CG sphere <input type="color" id="cg_sphere" value="{cg_sphere_color}"></label>
  <label>Sphere &alpha; <input type="range" id="cg_alpha" min="0" max="1" step="0.05" value="{cg_sphere_opacity}"> <span class="val" id="alpha_val">{cg_sphere_opacity}</span></label>
  <label>Sphere &times; <input type="range" id="cg_scale" min="4" max="20" step="0.5" value="{cg_sphere_scale}"> <span class="val" id="scale_val">{cg_sphere_scale}</span></label>
  <label>CG offset <input type="range" id="cg_offset" min="10" max="80" step="2" value="{cg_x_offset}"> <span class="val" id="offset_val">{cg_x_offset}</span> &Aring;</label>
</div>
<div class="viewer" id="view"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
(function() {{
  var v = $3Dmol.createViewer("view", {{ backgroundColor: "{bg_color}" }});

  var aaPdb  = `{aa_pdb}`;
  var cgOrig = `{cg_pdb}`;  // original, no offset

  var beadRadii = {bead_radii_json};
  var beadTypes = {bead_types_json};

  function shiftPdb(pdb, dx) {{
    return pdb.replace(/^(ATOM  |HETATM)(.{{24}})(.{{8}})/gm,
      function(m, pre, mid, xStr) {{
        var x = parseFloat(xStr) + dx;
        return pre + mid + x.toFixed(3).padStart(8);
      }});
  }}

  function rebuild() {{
    var cgStick  = document.getElementById("cg_stick").value;
    var cgColor  = document.getElementById("cg_sphere").value;
    var alpha    = parseFloat(document.getElementById("cg_alpha").value);
    var scale    = parseFloat(document.getElementById("cg_scale").value);
    var offset   = parseFloat(document.getElementById("cg_offset").value);
    document.getElementById("alpha_val").textContent  = alpha.toFixed(2);
    document.getElementById("scale_val").textContent  = scale;
    document.getElementById("offset_val").textContent = offset;

    v.removeAllModels();
    v.removeAllShapes();

    v.addModel(aaPdb, "pdb");
    v.setStyle({{ model: 0 }}, {{ stick: {{}} }});                // AA default

    v.addModel(shiftPdb(cgOrig, offset), "pdb");
    v.setStyle({{ model: 1 }}, {{ stick: {{ color: cgStick }} }}); // same thickness as AA

    var atoms = v.getModel(1).selectedAtoms({{}});
    for (var i = 0; i < atoms.length; i++) {{
      var a = atoms[i];
      var r = (beadRadii[beadTypes[i]] || 0.235) * scale;
      v.addSphere({{ center: {{ x: a.x, y: a.y, z: a.z }}, radius: r, color: cgColor, alpha: alpha }});
    }}
    v.zoomTo();
    v.render();
  }}

  ["cg_stick","cg_sphere","cg_alpha","cg_scale","cg_offset"].forEach(function(id) {{
    document.getElementById(id).addEventListener("input", rebuild);
  }});

  rebuild();
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

    # Per-atom bead types for addSphere radii
    _, bead_types = _parse_itp(itp_path)
    bead_radii = {bt: _bead_radius(bt) for bt in set(bead_types)}

    html = _HTML_TEMPLATE.format(
        title=f"{ligand_name} &mdash; AA vs CG",
        bg_color=BG_COLOR,
        aa_pdb=aa_pdb.read_text(),
        cg_pdb=cg_pdb.read_text(),                       # no offset — JS handles it
        cg_stick_color=CG_STICK_COLOR,
        cg_sphere_color=CG_SPHERE_COLOR,
        cg_sphere_opacity=CG_SPHERE_OPACITY,
        cg_sphere_scale=CG_SPHERE_SCALE,
        cg_x_offset=CG_X_OFFSET,
        bead_radii_json=json.dumps(bead_radii),
        bead_types_json=json.dumps(bead_types),
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
