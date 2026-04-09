The kernels will replace lib.rs

# kernel: vpm48_engine.rs
python3 TEST_v3_4.py

# kernel: vpm48_engine_optimized.rs
python3 -u test_vpm48_v3_8.py --particle higgs --residue 0
python3 -u test_vpm48_v3_8.py --particle z --residue 0 --residue 24
python3 -u test_vpm48_v3_8.py --particle w

# Kernel vpm48_engine_optimized3.rs
test_vcv48_full_v2.py

# kernel: vpm48_engine_top.rs
python3 -u TOP_VPM48.py

# kernel: oh_group2
TEST_OH_v2.py
TEST_OH.py

# Kernel: vcv48
HUNTER_GRAVITON_v2.py
HUNTER_GRAVITON.py
AXION_ULA.py --mode full --nf-min 1 --nf-max 100000 --residues 24 --step 48



>> sudo swapoff /swapfile
>> sudo rm -f /swapfile
>> sudo dd if=/dev/zero of=/swapfile bs=1M count=10240 status=progress 
>> sudo mkswap /swapfile
>> sudo swapon /swapfile
>> free -h 



Vallejos, O. A. (2026). Vacuum Crystallography (Vitrum Cosmicum Vacui (VCV48)). (vA). Zenodo. 
https://doi.org/10.5281/zenodo.19091908


Vallejos, O. A. (2026). Vacuum Crystallography (Vitrum Cosmicum Vacui (VCV48)). (vB). Zenodo. 
https://doi.org/10.5281/zenodo.19094323

Vallejos, O. A. (2026). Vacuum Crystallography (Vitrum Cosmicum Vacui (VCV48)). (vC). Zenodo. https://doi.org/10.5281/zenodo.19095180


Vallejos, O. A. (2026). Vitrum Cosmicum Vacui (VCV4848). (Annex I). Zenodo. https://doi.org/10.5281/zenodo.19487810
