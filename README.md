AI²MD (Artificial Intelligence for Accelerated Molecular Dynamics) is a hybrid framework that integrates ab-initio molecular dynamics (MD) with Machine Learning Interatomic Potentials (MLIPs) to enable fast and chemically accurate simulations of complex reactive systems, such as plasma-surface interactions and high-temperature decomposition.

🚀 1. Accelerated Dynamics via MLIP
	•	Replace traditional DFT-based forces with MLIP models like ORBNet, which drastically reduce computational costs while maintaining high accuracy.
	•	Run high-temperature simulations (e.g., 2000 K) with millions of steps using neural network-based potentials trained on high-level quantum data.

🔍 2. Real-Time Chemical Analysis
	•	Automatically extract molecular species from XYZ trajectories using bonding heuristics based on UFF radii.
	•	Use depth-first search (DFS) to cluster atoms into molecular fragments and track their evolution across time.
