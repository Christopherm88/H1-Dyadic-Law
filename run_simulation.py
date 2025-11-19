from h1_dyadic_law import simulate_dyad
import numpy as np
import json
from pathlib import Path

rng = np.random.default_rng(42)
n_dyads = 500
trajectories = []

for i in range(n_dyads):
    traj = simulate_dyad(
        S0_i=rng.normal(4.5, 1.2),
        S0_j=rng.normal(4.5, 1.2),
        V_i=rng.normal(0.5, 0.2),
        V_j=rng.normal(0.5, 0.2),
        tau_i=rng.normal(600, 300),
        tau_j=rng.normal(600, 300),
        rng=rng
    )
    trajectories.append(traj)
    if (i+1) % 50 == 0:
        print(f"Completed {i+1}/{n_dyads} dyads")

final_dfs = [t[-1]["∆F"] for t in trajectories]
print(f"\nMedian final ∆F = {np.median(final_dfs):.4f}")
print(f"Convergence (<0.05): {100*np.mean(np.array(final_dfs)<0.05):.1f}%")

Path("output").mkdir(exist_ok=True)
with open("output/sample_trajectory.json", "w") as f:
    json.dump(trajectories[0], f, indent=2)
print("Sample trajectory saved to output/sample_trajectory.json")
