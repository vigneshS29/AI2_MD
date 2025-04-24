import argparse
from collections import defaultdict, Counter
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt

# --- UFF Bond Radii Dictionary ---                                                                                                                                                                                                                                                                                                                                                                                                          
UFF_RADII = {
    'H': 0.354, 'He': 0.849, 'Li': 1.336, 'Be': 1.074, 'B': 0.838, 'C': 0.757, 'N': 0.700,
    'O': 0.658, 'F': 0.668, 'Ne': 0.920, 'Na': 1.539, 'Mg': 1.421, 'Al': 1.244, 'Si': 1.117,
    'P': 1.117, 'S': 1.064, 'Cl': 1.044, 'Ar': 1.032, 'K': 1.953, 'Ca': 1.761, 'Sc': 1.513,
    'Ti': 1.412, 'V': 1.402, 'Cr': 1.345, 'Mn': 1.382, 'Fe': 1.335, 'Co': 1.241, 'Ni': 1.164,
    'Cu': 1.302, 'Zn': 1.193, 'Ga': 1.260, 'Ge': 1.197, 'As': 1.211, 'Se': 1.190, 'Br': 1.192,
    'Kr': 1.147, 'Rb': 2.260, 'Sr': 2.052, 'Y': 1.698, 'Zr': 1.564, 'Nb': 1.473, 'Mo': 1.484,
    'Tc': 1.322, 'Ru': 1.478, 'Rh': 1.332, 'Pd': 1.338, 'Ag': 1.386, 'Cd': 1.403, 'In': 1.459,
    'Sn': 1.398, 'Sb': 1.407, 'Te': 1.386, 'I': 1.382, 'Xe': 1.267, 'Cs': 2.570, 'Ba': 2.277,
    'La': 1.943, 'Hf': 1.611, 'Ta': 1.511, 'W': 1.526, 'Re': 1.372, 'Os': 1.372, 'Ir': 1.371,
    'Pt': 1.364, 'Au': 1.262, 'Hg': 1.340, 'Tl': 1.518, 'Pb': 1.459, 'Bi': 1.512, 'Po': 1.500,
    'At': 1.545, 'Rn': 1.420, 'default': 0.7
}

def get_radius(symbol):
    return UFF_RADII.get(symbol, UFF_RADII["default"])

def get_bonds(atoms, tolerance=0.4):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            r_i = get_radius(symbols[i])
            r_j = get_radius(symbols[j])
            cutoff = r_i + r_j + tolerance
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < cutoff:
                bonds.append((i, j))
    return bonds

def get_molecule_clusters(atoms, bonds):
    graph = defaultdict(list)
    for i, j in bonds:
        graph[i].append(j)
        graph[j].append(i)

    visited = set()
    molecules = []

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for i in range(len(atoms)):
        if i not in visited:
            group = []
            dfs(i, group)
            molecules.append(atoms[group])

    return molecules

def get_formula(mol):
    symbols = sorted(mol.get_chemical_symbols())
    return ''.join(f"{el}{symbols.count(el)}" for el in sorted(set(symbols)))

def compute_species_fractions(xyz_file, tolerance=0.4, timestep_fs=1.0):
    frames = read(xyz_file, index=":")
    species_set = set()
    fractions_list = []
    time_axis = []

    for i, atoms in enumerate(frames):
        bonds = get_bonds(atoms, tolerance)
        mols = get_molecule_clusters(atoms, bonds)
        counts = Counter(get_formula(mol) for mol in mols)
        total = sum(counts.values())
        fractions = {k: v / total for k, v in counts.items()}
        species_set.update(fractions.keys())
        fractions_list.append(fractions)
        time_axis.append(i * timestep_fs * 1e-3)  # fs to ps                                                                                                                                                                                                                                                                                                                                                                                 

        # Print frame information                                                                                                                                                                                                                                                                                                                                                                                                            
        print(f"\nFrame {i} (Time {time_axis[-1]:.2f} ps):")
        for species, count in counts.items():
            print(f"  {species}: {count}")

    return time_axis, fractions_list, sorted(species_set)

def plot_species_fractions(time_axis, fractions_list, species_list, output="result.png"):
    for species in species_list:
        y = [frame.get(species, 0.0) for frame in fractions_list]
        plt.plot(time_axis, y, label=species)

    plt.xlabel("Time (ps)")
    plt.ylabel("Molecule Fraction")
    plt.title("Species vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz", help="Input XYZ trajectory file")
    parser.add_argument("--timestep", type=float, default=1.0, help="Time per frame in fs")
    parser.add_argument("--tolerance", type=float, default=0.4, help="Bond tolerance (Å)")
    parser.add_argument("--output", default="result.png", help="Output plot file name")
    args = parser.parse_args()

    time_axis, fractions, species = compute_species_fractions(
        args.xyz, tolerance=args.tolerance, timestep_fs=args.timestep
    )
    plot_species_fractions(time_axis, fractions, species, output=args.output)

if __name__ == "__main__":
    main()

