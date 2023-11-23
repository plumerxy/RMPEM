from os.path import dirname, basename, isfile, join
import glob

# Import new models here
from .PG_paper import GraphGCN
from .GNN_paper import GraphGCN
from .GNN_paper import GraphGCN_inter

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
