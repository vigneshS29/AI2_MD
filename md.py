import argparse
import os
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")
    return device


def initialize_atoms(input_file, cell_size):
    atoms = read(input_file)
    atoms.set_cell([cell_size] * 3)
    atoms.set_pbc([True] * 3)
    return atoms


def prepare_output_paths(input_file, output_dir, suffix):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    suffix_str = f"_{suffix}" if suffix else ""
    xyz_path = os.path.join(output_dir, f"{base_name}{suffix_str}_MD.xyz")
    log_path = os.path.join(output_dir, f"{base_name}{suffix_str}_md.log")
    return xyz_path, log_path


def run_md(atoms, xyz_path, log_path, temperature_K, timestep, friction, steps, traj_interval, log_interval):
    dyn = Langevin(atoms, timestep, temperature_K=temperature_K, friction=friction)
    dyn.attach(lambda: write(xyz_path, atoms, append=True), interval=traj_interval)
    dyn.attach(MDLogger(dyn, atoms, log_path), interval=log_interval)
    dyn.run(steps)


def run_md_simulation(
    input_file,
    output_dir,
    suffix="",
    cell_size=30.00,
    temperature_K=500,
    timestep=0.5 * units.fs,
    friction=0.01 / units.fs,
    equil_steps=1,
    prod_steps=100000000,
    traj_interval=20,
    log_interval=1
):
    device = setup_device()

    # --- Equilibration ---                                                                                                                                                                                                                                                                                                                                                                                                                  
    atoms = initialize_atoms(input_file, cell_size)
    atoms.calc = ORBCalculator(pretrained.orb_d3_v2(), device=device)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    eq_xyz, eq_log = prepare_output_paths(input_file, output_dir, f"{suffix}_equil")
    print(f"[INFO] Starting equilibration MD: {input_file} -> {eq_xyz}")
    run_md(
        atoms, eq_xyz, eq_log,
        300, timestep, friction,
        steps=equil_steps,
        traj_interval=traj_interval,
        log_interval=log_interval
    )
    print("[INFO] Equilibration completed.")

    # --- Save last equil frame ---                                                                                                                                                                                                                                                                                                                                                                                                          
    last_equil_frame = read(eq_xyz, index=-1)
    last_equil_path = os.path.join(output_dir, f"{suffix}_last_equil_frame.xyz")
    write(last_equil_path, last_equil_frame)
    print(f"[INFO] Last equilibration frame saved to: {last_equil_path}")

    # --- Production ---                                                                                                                                                                                                                                                                                                                                                                                                                     
    atoms = initialize_atoms(last_equil_path, cell_size)
    atoms.calc = ORBCalculator(pretrained.orb_d3_v2(), device=device)

    prod_xyz, prod_log = prepare_output_paths(input_file, output_dir, f"{suffix}_prod")
    print(f"[INFO] Starting production MD: {last_equil_path} -> {prod_xyz}")
    run_md(
        atoms, prod_xyz, prod_log,
        temperature_K, timestep, friction,
        steps=prod_steps,
        traj_interval=traj_interval,
        log_interval=log_interval
    )
    print("[INFO] Production MD simulation completed.")


def main():
    parser = argparse.ArgumentParser(description="Run equilibration and production MD with ORB.")
    parser.add_argument("input_file", help="Input XYZ file")
    parser.add_argument("output_dir", help="Directory to store output files")
    parser.add_argument("--suffix", default="", help="Optional identifier for output files")
    parser.add_argument("--equil_steps", type=int, default=20, help="Equilibration steps")
    parser.add_argument("--prod_steps", type=int, default=1000000, help="Production steps")

    args = parser.parse_args()

    run_md_simulation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        suffix=args.suffix,
        equil_steps=args.equil_steps,
        prod_steps=args.prod_steps
    )


if __name__ == "__main__":
    main()
