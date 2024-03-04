# sdf_to_props.py

__doc__ = """Calculates a set of chemical descriptors related to the overall
shape of molecules.
"""

###############################################################################
import numpy as np
import pybel
# import openbabel
import optparse
import json
import sys
from main import get_predicted_properties
###############################################################################


def main():
    # usage = """
    # usage: %prog [options]
    # """
    # parser = optparse.OptionParser(usage)

    smiles = sys.argv[1]
    print(smiles)
    predictions = get_predicted_properties(smiles)

    # process predictions 
    # ...
        
    data = {}
    data['molecule'] = {
        'formal_chrg' : 1,
        'ppsa' : 1,
        'hbdsa' : sys.argv[1]
    }
    json.dump(data, sys.stdout)

if __name__=='__main__':
	main()
