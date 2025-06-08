The files in this directory are used to improve reproducibility. 

You should move them out of the `./scripts` directory and run them in the root directory. Or you can run them through `python ./scripts/....py` after setting the environment variable `PYTHONPATH=./`.

The following is a brief description of the functions of each file. You can get detailed configuration information by running `python ./scripts/....py --help`:
- `search_unknown_network.py`: Simultaneously recover network structure and dynamics formulas when network structure is unknown.
- `search hetero.py`: Recover the dynamics formulas for each class on a heterogeneous graph.
- `search_hidden_edge_weight.py`: Simultaneously recover edge weights and dynamics formulas when there are hidden edge weights.
- `search_bacteria.py`: Search for ecological community dynamics.
- `search_gene.py`: Search for gene regulation dynamics.
- `search_epidemic.py`: Search for epidemic spreading dynamics.
- `COVID19_{CHI,CN,ILS,NYC,NYS,US,World}.ipynb`: Download epidemic spreading data in different regions.
- `baseline_TwoPhase.py`: Identifying network dynamics formulas in synthetic data using the two-phase approach (Gao and Yan, NCS'22).
- `baseline_TwoPhase.sh`: Batch identification of network dynamics formulas in synthetic data using the two-phase approach.
- `baseline_TwoPhase2.sh`: Testing the robustness of the two-phase approach to identifying network dynamics formulas in synthetic data.
- `baseline_MPNN.py`: Fitting genetic and population data using message-passing graph neural network (MPNN).
- `baseline_NDCN.py`: Fitting genetic and population data using graph neural ODE network (NDCN).
- `baseline_MPNN_gs.sh`: Using grid search to determine the optimal hyperparameters in MPNN.
- `baseline_NDCN_gs.sh`: Using grid search to determine the optimal hyperparameters in NDCN.
