# Best-Arm-Identification-with-Feedback-Graph

Best Arm Identification with Feedback Graphs. Please, refer to the main paper for more details.

To run the experiments, see instructions below.


## Libraries and optimizers

- **Language**: Python 3.10.12
- **Libraries**: The following libraries are required: `pandas, numpy, tqdm, scipy, matplotlib, cvxpy, seaborn, latex, multipledispatch, jupyter`. For a complete list, check also the `requirements.txt` file. We suggest to create a `venv` environment, activate it and install the libraries using the command `python -m pip install -r requirements.txt`
- **Optimizers**: We suggest to use the GUROBI optimizer. The original figures were obtained using GUROBI 10. To change the optimizer, change the solver in `baifg\utils\characteristic_time.py`
whenever the function `cp.solve(solver=cp.GUROBI)` is called. We suggest using CLARABEL (`cp.CLARABEL`) as an alternative. 

Make sure to have latex installed to properly plot the figures (e.g. MikTEX).


## Example of symmetric feedback graph
To evaluate the example of a symmetric feedback graph please refer to the notebook `example_symmetric_graph.ipynb`.

## Example of the loopy star graphs
To evaluate the example of a symmetric feedback graph please refer to the notebook `example_loopystar.ipynb`.

For the hard example refer to the notebook `example_loopystar_hard.ipynb`.


## Example of the loopless clique graph
To evaluate the example of a symmetric feedback graph please refer to the notebook `example_looplessclique.ipynb`.

## Example of the ring graph
To evaluate the example of a symmetric feedback graph please refer to the notebook `example_rin.ipynb`.

## To run the simulations and plot the main numerical results
To run the main simulations, refer to the python file `run_experiments.py`.
Adjust the variables:
- `NUM_PROCESSES`, which indicates how many processes to use to run the simulations. 
- `Nsims`, which defines how many seeds to use

All the data will be saved in the `data` folder, within a folder with the current date
in the format `%Y-%m-%d-%H-%M`. All the data will be saved in a lzma file named `full_data.lzma`.

To plot the results, run the notebook `plot_data.ipynb`. Set the `path = "data/{FOLDER}/full_data.lzma"` variable at the beginning to point to the correct folder.