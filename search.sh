python search.py --data ./data/synthetic/KUR.json --vars x omega --name Kuramoto --seed 0
python search.py --data ./data/synthetic/FHN.json --vars x y --name FitzHugh-Nagumo --seed 0
python search.py --data ./data/synthetic/CR.json --vars x y z omega --name Coupled-Rössler --seed 0
python search.py --data ./data/synthetic/HCR.json --vars x y z --name Homogeneous-Coupled-Rössler --seed 0
python search.py --data ./data/synthetic/WC.json --vars x --name Wilson-Cowan --seed 0
python search.py --data ./data/synthetic/MM.json --vars x --name Michaelis-Menten --seed 0
python search.py --data ./data/synthetic/LV.json --vars x alpha theta --name Lotka-Volterra --seed 0
python search.py --data ./data/synthetic/MP.json --vars x alpha theta --name Mutualistic-Population --seed 0
python search.py --data ./data/synthetic/SIS.json --vars x delta --name Susceptible-Infected-Susceptible --seed 0
python search.py --data ./data/synthetic/GR.json --vars x --name Gene-Regulatory --seed 0

