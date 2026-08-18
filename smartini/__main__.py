import argparse
import logging
import sys

from . import __version__, solver
from .common import *
from .topology import gen_molecule_sdf, gen_molecule_smi

def checkArgs(args):
    """ AutoM3 change: removed argument --top for simpler input, .itp file will be named by using --mol argument (mol.itp)"""

    if not args.sdf and not args.smi:
        parser.error("run requires --sdf or --smi")

    if not args.molname:
        parser.error("run requires --mol")

parser = argparse.ArgumentParser(
    prog="smartini",
    description="Generates Martini 3 force field for atomistic structures of small organic molecules",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Developers:\n===========\nMagdalena Szczuka (magdalena.szczuka [at] univ-tlse3.fr)\nTristan Bereau (bereau [at] mpip-mainz.mpg.de)\nKiran Kanekal (kanekal [at] mpip-mainz.mpg.de)
Andrew Abi-Mansour (andrew.gaam [at] gmail.com)""",
)
parser.add_argument(
    "--mode", type=str, choices=["run"], default="run", help="mode: run (compute FF)"
)

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument(
    "--sdf",
    dest="sdf",
    type=str,
    required=False,
    help="SDF file of atomistic coordinates",
)

group.add_argument(
    "--smi",
    dest="smi",
    type=str,
    required=False,
    help="SMILES string of atomistic structure",
)
parser.add_argument("--logp",dest="logp",type=str,required=None,help="File with partial smiles and associated logP") #AutoM3 change: for custom database of fragments and logp, default file in scripts directory (logP_smi.dat)
parser.add_argument("--mol", dest="molname", type=str, required=False, help="Name of CG molecule")
parser.add_argument("--aa", dest="aa", type=str, required=False, help="filename of all-atom structure .gro file")
parser.add_argument("-v","--verbose",dest="verbose",action="count",default=0, required=False,help="increase verbosity")
parser.add_argument("--fpred",dest="forcepred",action="store_true", required=False,help="Atomic partitioning prediction")
parser.add_argument("--simple",dest="simple_model",action="store_true",required=False,help="Simple model without dihedrals nor virtual sites") #AutoM3 change
parser.add_argument("--canon",dest="canonic_smiles",action="store_true",required=False,help="Translate to RdKit canon structure") #AutoM3 change

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

checkArgs(args)

if args.verbose >= 2:
    level = logging.DEBUG
elif args.verbose >= 1:
    level = logging.INFO
else:
    level = logging.WARNING

logging.basicConfig(
    filename="smartini.log",
    format="%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d): %(message)s",
    level=level,
)

logger = logging.getLogger(__name__)

logger.info("Running smartini v{}".format(__version__))

# Generate molecule's structure from SDF or SMILES
if args.sdf:
    mol = gen_molecule_sdf(args.sdf)
    if args.canonic_smiles: smiles = str(Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=False)))
    else : smiles = str(Chem.MolToSmiles(mol, isomericSmiles=False))
else:
    if args.canonic_smiles: smiles = (Chem.CanonSmiles(args.smi))
    else : smiles = args.smi 
    mol, _ = gen_molecule_smi(smiles)
    
### AutoM3 change ###
topname=args.molname+".itp"
groname=args.molname+".gro"
bartenderfname=""

if args.bartender_output:
    bartenderfname=args.molname+"_bartender.inp"
    cg = solver.Cg_molecule(mol, smiles, args.molname, args.simple_model, topname, bartenderfname, args.bartender_output, args.logp, args.forcepred)
else:
    cg = solver.Cg_molecule(mol, smiles, args.molname, args.simple_model, topname, bartenderfname, args.bartender_output, args.logp, args.forcepred)
if args.aa:
    cg.output_aa(args.aa)
cg.output_cg(groname)