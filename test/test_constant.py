import sys, os
sys.path.append(os.path.abspath('./'))
print(sys.path)

import demeter as dm

def test_constant():
    print(f"ROOT DIRECTORY: {dm.ROOT_DIRECTORY}")
    assert isinstance(dm.ROOT_DIRECTORY, str)
    assert isinstance(dm.OPTIM_SAVE_DIR, str)
    assert isinstance(dm.FIELD_TO_SAVE, list)
