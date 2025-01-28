import sys, os
sys.path.append(os.path.abspath('./'))
print(sys.path)

import demeter as dm

def test_constant():
    print(f"ROOT DIRECTORY: {dm.ROOT_DIRECTORY}")
    assert isinstance(dm.ROOT_DIRECTORY, str)
    assert isinstance(dm.OPTIM_SAVE_DIR, str)
    assert isinstance(dm.FIELD_TO_SAVE, list)

def test_at_build():
    print(f"os.path.abspath(__file__) : {os.path.abspath(__file__)}")
    print(f"ROOT DIRECTORY: {dm.ROOT_DIRECTORY}")

    print(f"in ROOT_DIRECTORY: {os.listdir(dm.ROOT_DIRECTORY)}")
    print(f"in .. ROOT_DIRECTORY: {os.listdir(os.path.join(dm.ROOT_DIRECTORY,'..'))}")
    assert os.path.exists(dm.ROOT_DIRECTORY)
    assert os.path.exists(dm.OPTIM_SAVE_DIR), \
        f"OPTIM_SAVE_DIR: {dm.OPTIM_SAVE_DIR}"
    assert os.path.exists(dm.ROOT_DIRECTORY+"/examples/")

