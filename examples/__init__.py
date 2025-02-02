import sys, os
# add the parent directory to the path
base_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(base_path,'src'))