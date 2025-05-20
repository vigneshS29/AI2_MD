import os
import argparse
import torch
import torch._dynamo

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from fairchem.core import FAIRChemCalculator
from mace.calculators import mace_mp

import logging
import numpy as np

torch._dynamo.config.suppress_errors = True

def infer_cell(atoms, padding=2.0):
    if atoms.get_cell().volume < 1e-3:
        pos = atoms.get_positions()
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        cell = maxs - mins + 2 * padding
        atoms.translate(-mins + padding)
        atoms.set_cell(cell)
        atoms.set_pbc([True, True, True])
    return atoms

def setup_calc(model, device):
    if model == "orb":
        orbff = pretrained.orb_d3_v2()
        return ORBCalculator(orbff, device=device)
    elif model == "fair":
        return FAIRChemCalculator(hf_hub_filename="uma_sm.pt", device=device.type, task_name="omat")
    elif model == "mace":
        return mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device.type)
    else:
        raise ValueError(f"Unsupported model: {model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file")
    parser.add_argument("--model", choices=["orb", "fair", "mace"], default="orb")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temp", type=float, default=300)
    parser.add_argument("--timestep_fs", type=float, default=0.5)
    parser.add_argument("--output", default="md_out.xyz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atoms = read(args.xyz_file)
    atoms = infer_cell(atoms)

    atoms.calc = setup_calc(args.model, device)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)

    dyn = Langevin(atoms, args.timestep_fs * units.fs, temperature_K=args.temp, friction=0.01 / units.fs)
    dyn.attach(lambda: write(args.output, atoms.copy(), append=True), interval=10)
    dyn.attach(MDLogger(dyn, atoms, "md.log"), interval=1)
    dyn.run(args.steps)

if __name__ == "__main__":
    main()
    
