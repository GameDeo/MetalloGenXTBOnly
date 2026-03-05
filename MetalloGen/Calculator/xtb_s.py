import os
import distutils.spawn
import numpy as np
from copy import deepcopy

### ace-reaction libraries ###
from MetalloGen import chem, process

def parse_xtb_energy(log_file):
    """Parse the standalone xtb stdout log for the final energy."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    energy = None
    for line in reversed(lines):
        if "| TOTAL ENERGY" in line:
            energy_line = line.strip().split()
            try:
                energy = float(energy_line[3])
                break
            except ValueError:
                continue
    return energy

def parse_xtb_trj(xyz_file):
    """Parse a standard multi-frame XYZ trajectory (like xtbopt.xyz or xtbirc.xyz)."""
    try:
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    trajectory = []
    index = 0
    while index < len(lines):
        try:
            num_atom = int(lines[index].strip())
            index += 1
            
            comment_line = lines[index].strip()
            try:
                energy = float(comment_line.split()[1]) if 'energy:' in comment_line else 0.0
            except:
                energy = 0.0
            index += 1
            
            atom_list = []
            for _ in range(num_atom):
                atom_info = lines[index].strip().split()
                element = atom_info[0]
                x, y, z = float(atom_info[1]), float(atom_info[2]), float(atom_info[3])
                
                atom = chem.Atom(element)
                atom.x = x
                atom.y = y
                atom.z = z
                atom_list.append(atom)
                index += 1
                
            if atom_list:
                molecule = chem.Molecule()
                molecule.atom_list = atom_list
                molecule.energy = energy
                trajectory.append(molecule)
        except Exception:
            break
            
    return trajectory

def parse_xtb_gradient(grad_file, num_atoms):
    """Parse xTB's gradient file (outputs in Hartree/Bohr)."""
    try:
        with open(grad_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return np.zeros((num_atoms, 3))
    
    # Line 0: $coord, Line 1: energy/etc, Line 2 to 2+N-1: coords, then $grad, then gradients
    grad_start = 0
    for i, line in enumerate(lines):
        if '$grad' in line:
            grad_start = i + 1
            break
            
    force = []
    for i in range(num_atoms):
        parts = lines[grad_start + i].strip().split()
        force.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
    return -np.array(force)

def parse_xtb_hessian(hess_file, num_atoms):
    """Parse xTB's hessian file (outputs in Hartree/Bohr^2)."""
    try:
        with open(hess_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return np.zeros((3*num_atoms, 3*num_atoms))
    
    hessian = []
    start_idx = 1 if '$hessian' in lines[0] else 0
    for line in lines[start_idx:]:
        if line.strip().startswith('$'): continue
        parts = line.strip().split()
        hessian.extend([float(x) for x in parts])
        
    hess_matrix = np.array(hessian).reshape((3*num_atoms, 3*num_atoms))
    return hess_matrix


class XTB_Standalone:
    
    def __init__(self, command='xtb', working_directory=None):
        check = distutils.spawn.find_executable(command)
        self.name = 'xtb'
        if check is None:
            print(f'{command} not found in system PATH!')
            exit()
            
        self.command = command
        self.energy_unit = 'Hartree'
        
        if working_directory is not None:
            if not os.path.exists(working_directory):
                os.makedirs(working_directory, exist_ok=True)
                
        if working_directory is None:
            working_directory = os.getcwd()

        self.working_directory = working_directory
        self.error_directory = None

    def __str__(self):
        content = f'working_directory: {self.working_directory}\n'
        content += f'command: {self.command}\n'
        content += f'Energy: {self.energy_unit}\n'
        return content

    def get_default_mol_params(self, molecule):
        try:
            chg = molecule.get_chg()
        except:
            chg = 0
        try:
            e_list = molecule.get_num_of_lone_pair_list()
            num_of_unpaired_e = len(np.where((2*e_list) % 2 == 1)[0])
            multiplicity = num_of_unpaired_e + 1
        except:
            z_sum = np.sum(molecule.get_z_list())
            multiplicity = (z_sum - chg) % 2 + 1
        return chg, multiplicity

    def make_input(self, molecules, file_name='test'):
        """Writes a standard .xyz file. Supports multi-structure for TS searches."""
        inpstring = ""
        for molecule in molecules:
            if molecule is None: continue
            num_atoms = len(molecule.atom_list)
            inpstring += f"{num_atoms}\nGenerated by XTB_Standalone\n"
            for atom in molecule.atom_list:
                inpstring += f"{atom.element:<4} {atom.x:>12.6f} {atom.y:>12.6f} {atom.z:>12.6f}\n"

        with open(f'{file_name}.xyz', 'w') as f:
            f.write(inpstring)

    def write_constraints(self, constraints, file_name='xcontrol'):
        """Translates Gaussian Redundant Coordinates into xTB $constrain block."""
        if not constraints: return False
        
        content = "$constrain\nforce constant=1.0\n"
        atoms = []
        for c in constraints:
            if len(c) == 1: # Freeze atom
                atoms.append(str(c[0]+1))
            elif len(c) == 2: # Freeze bond
                content += f"distance: {c[0]+1}, {c[1]+1}, auto\n"
            elif len(c) == 3: # Freeze angle
                content += f"angle: {c[0]+1}, {c[1]+1}, {c[2]+1}, auto\n"
            elif len(c) == 4: # Freeze dihedral
                content += f"dihedral: {c[0]+1}, {c[1]+1}, {c[2]+1}, {c[3]+1}, auto\n"
        
        if atoms:
            content += f"atoms: {','.join(atoms)}\n"
        content += "$end\n"
        
        with open(file_name, 'w') as f:
            f.write(content)
        return True

    def move_file(self, file_name, save_directory):
        if file_name is not None and save_directory is not None:
            current_directory = os.getcwd()
            try:
                os.chdir(self.working_directory)
                os.makedirs(save_directory, exist_ok=True)
                os.system(f'mv {file_name}.* {save_directory}/ 2>/dev/null')
                os.system(f'mv xtb* {save_directory}/ 2>/dev/null')
                os.system(f'mv gradient hessian vibspectrum {save_directory}/ 2>/dev/null')
            except Exception as e:
                print(f"File move failed: {e}")
            finally:
                os.chdir(current_directory)

    def get_energy(self, molecule, chg=None, multiplicity=None, file_name='sp', extra='', save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(molecule)
        uhf = multiplicity - 1
        
        self.make_input([molecule], file_name=file_name)
        os.system(f'{self.command} {file_name}.xyz --chrg {chg} --uhf {uhf} {extra} > {file_name}.out')
        
        energy = parse_xtb_energy(f'{file_name}.out')
        self.move_file(file_name, save_directory)
        
        converter = 1.0
        if energy is None: return None
        if self.energy_unit == 'kcal': converter = 627.509
        elif self.energy_unit == 'Hartree': converter = 1.0 # xTB default is Eh
            
        os.chdir(current_directory)
        return converter * energy

    def get_force(self, molecule, chg=None, multiplicity=None, file_name='force', extra='', save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(molecule)
        uhf = multiplicity - 1
        
        self.make_input([molecule], file_name=file_name)
        os.system(f'{self.command} {file_name}.xyz --grad --chrg {chg} --uhf {uhf} {extra} > {file_name}.out')
        
        bohr_to_angstrom = 0.529177
        force = parse_xtb_gradient('gradient', len(molecule.atom_list))
        
        self.move_file(file_name, save_directory)
        os.chdir(current_directory)
        return force / bohr_to_angstrom

    def get_hessian(self, molecule, chg=None, multiplicity=None, file_name='hessian', extra='', save_directory=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(molecule)
        uhf = multiplicity - 1
        
        self.make_input([molecule], file_name=file_name)
        os.system(f'{self.command} {file_name}.xyz --hess --chrg {chg} --uhf {uhf} {extra} > {file_name}.out')
        
        bohr_to_angstrom = 0.529177
        force = parse_xtb_gradient('gradient', len(molecule.atom_list)) # xtb --hess also dumps grad
        hessian = parse_xtb_hessian('hessian', len(molecule.atom_list))
        
        self.move_file(file_name, save_directory)
        os.chdir(current_directory)
        return force / bohr_to_angstrom, hessian / bohr_to_angstrom**2

    def optimize_geometry(self, molecule, constraints={}, chg=None, multiplicity=None, file_name='opt', extra='', save_directory=None, initial_hessian=None):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(molecule)
        uhf = multiplicity - 1
        
        self.make_input([molecule], file_name=file_name)
        
        opt_extra = extra
        if self.write_constraints(constraints):
            opt_extra += " --input xcontrol"
            
        os.system(f'{self.command} {file_name}.xyz --opt --chrg {chg} --uhf {uhf} {opt_extra} > {file_name}.out')
        
        energy = parse_xtb_energy(f'{file_name}.out')
        trj = parse_xtb_trj('xtbopt.xyz')
        
        if len(trj) > 0:
            process.locate_molecule(molecule, trj[-1].get_coordinate_list(), False)
            
        converter = 1.0
        if self.energy_unit == 'kcal': converter = 627.509
        if energy is not None:
            molecule.energy = energy * converter
            
        self.move_file(file_name, save_directory)
        os.chdir(current_directory)
        
    def relax_geometry(self, molecule, constraints={}, chg=None, multiplicity=None, file_name='relax', num_relaxation=200, maximal_displacement=1000, save_directory=None, initial_hessian=None):
        extra = f" --cycles {num_relaxation}"
        self.optimize_geometry(molecule, constraints, chg, multiplicity, file_name, extra, save_directory, initial_hessian)

    def relax_geometry_steep(self, molecule, constraints={}, chg=None, multiplicity=None, file_name='relax_steep', num_relaxation=200, maximal_displacement=1000, save_directory=None, initial_hessian=None):
        extra = f" --cycles {num_relaxation} --strict" 
        self.optimize_geometry(molecule, constraints, chg, multiplicity, file_name, extra, save_directory, initial_hessian)

    def search_ts(self, molecules, chg=None, multiplicity=None, method='ts', file_name='ts', extra='', save_directory=None, check_frequency=True):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        idx = next((i for i, m in enumerate(molecules) if m is not None), None)
        if idx is None: return None, {}
        
        ts_molecule = molecules[idx].copy()
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(ts_molecule)
        uhf = multiplicity - 1

        if 'qst' in method:
            # xTB path requires multi-structure xyz (R, TS_guess, P)
            valid_mols = [m for m in molecules if m is not None]
            self.make_input(valid_mols, file_name=file_name)
            os.system(f'{self.command} {file_name}.xyz --path --chrg {chg} --uhf {uhf} {extra} > {file_name}.out')
            trj = parse_xtb_trj('xtbpath.xyz') # Extract highest energy frame
            if trj: ts_molecule = sorted(trj, key=lambda m: m.energy)[-1]
                
        else: # Standard TS via ohess
            self.make_input([ts_molecule], file_name=file_name)
            os.system(f'{self.command} {file_name}.xyz --ohess --chrg {chg} --uhf {uhf} {extra} > {file_name}.out')
            trj = parse_xtb_trj('xtbopt.xyz')
            if trj: process.locate_molecule(ts_molecule, trj[-1].get_coordinate_list(), False)
                
        energy = parse_xtb_energy(f'{file_name}.out')
        converter = 627.509 if self.energy_unit == 'kcal' else 1.0
        if energy: ts_molecule.energy = energy * converter

        imaginary_vibrations = {}
        if check_frequency:
            # xTB dumps frequencies in 'vibspectrum'
            try:
                with open('vibspectrum', 'r') as f:
                    lines = f.readlines()
                    for line in lines[2:]: # Skip header
                        parts = line.strip().split()
                        freq = float(parts[0])
                        if freq < 0:
                            imaginary_vibrations[freq] = [] # xTB requires separate parser for normal modes
            except: pass
                
        self.move_file(file_name, save_directory)
        os.chdir(current_directory)
        return ts_molecule, imaginary_vibrations

    def run_irc(self, ts_molecule, chg=None, multiplicity=None, file_name='irc', extra='', save_directory=None, chkpoint_file='', params=dict()):
        current_directory = os.getcwd()
        os.chdir(self.working_directory)
        
        if chg is None or multiplicity is None:
            chg, multiplicity = self.get_default_mol_params(ts_molecule)
        uhf = multiplicity - 1
        
        self.make_input([ts_molecule], file_name=file_name)
        os.system(f'{self.command} {file_name}.xyz --irc --chrg {chg} --uhf {uhf} {extra} > {file_name}.out')
        
        irc_trajectory = parse_xtb_trj('xtbirc.xyz')
        converter = 627.509 if self.energy_unit == 'kcal' else 1.0
        
        for m in irc_trajectory:
            m.energy *= converter
            
        self.move_file(file_name, save_directory)
        os.chdir(current_directory)
        return irc_trajectory

    def clean_scratch(self, file_name='test.xyz'):
        os.system(f'rm -f {self.working_directory}/*.tmp {self.working_directory}/wbo')

    def change_working_directory(self, new_directory):
        """Updates the working directory and creates it if it doesn't exist."""
        if not os.path.exists(new_directory):
            os.makedirs(new_directory, exist_ok=True)
        self.working_directory = new_directory