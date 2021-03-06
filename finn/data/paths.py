"""
Paths to demo data used in the demo applications and the testing (whenever necessary)
"""

import pathlib

fct_sfc_data = str(pathlib.Path(__file__).parent.absolute()) + "/functionality/sfc/demo_data.npy"

fct_brainvision_data = str(pathlib.Path(__file__).parent.absolute()) + "/functionality/brainvision/data.vhdr"

per_fir_data = str(pathlib.Path(__file__).parent.absolute()) + "/performance/sfc/data_0.npy"

per_sfc_data_0 = str(pathlib.Path(__file__).parent.absolute()) + "/performance/sfc/data_0.npy"
per_sfc_data_1 = str(pathlib.Path(__file__).parent.absolute()) + "/performance/sfc/data_1.npy"
per_sfc_data_2 = str(pathlib.Path(__file__).parent.absolute()) + "/performance/sfc/data_2.npy"

per_pool_data = str(pathlib.Path(__file__).parent.absolute()) + "/performance/pool/data_0.npy"
