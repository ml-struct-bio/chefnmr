import multiprocessing
import numpy as np
from functools import partial
from typing import List, Optional, Dict, Any
from collections import Counter
from tqdm import tqdm

from .workers import (process_chunk_sigma_data, process_chunk_max_n_atoms, 
                     process_chunk_n_atoms_distribution, process_chunk_atom_types, 
                     process_chunk_solvents)


class ComputationManager:
    """Manages computation-heavy operations for the molecule factory."""
    
    def __init__(self, factory):
        """Initialize with reference to the factory."""
        self.factory = factory
    
    def compute_sigma_data(self, mol_indices: List[int], num_workers: Optional[int] = None) -> float:
        """Compute sigma data using multiprocessing."""
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, len(mol_indices))
        
        chunk_size = len(mol_indices) // num_workers + (1 if len(mol_indices) % num_workers else 0)
        chunks = [mol_indices[i:i+chunk_size] for i in range(0, len(mol_indices), chunk_size)]
        
        worker_fn = partial(
            process_chunk_sigma_data,
            factory_datadir=self.factory.datadir,
            factory_mode='r',
            factory_swmr=True,
            factory_db_path=getattr(self.factory.mol_idx_mapper, 'db_path', None)
        )
        
        print(f"Computing sigma_data using {num_workers} processes...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(worker_fn, chunks), total=len(chunks), desc="Computing sigma_data"))
        
        flattened_coords = np.concatenate(results)
        if flattened_coords.size == 0:
            print("No atom coordinates found.")
            return 0.0
        
        sigma_data = np.std(flattened_coords)
        print(f"Computed sigma_data = {sigma_data:.4f} from {flattened_coords.size} coordinate values")
        return float(sigma_data)
    
    def compute_max_n_atoms(self, mol_indices: List[int], num_workers: Optional[int] = None, remove_h: bool = False) -> int:
        """Compute maximum number of atoms using multiprocessing."""
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, len(mol_indices))
        
        chunk_size = len(mol_indices) // num_workers + (1 if len(mol_indices) % num_workers else 0)
        chunks = [mol_indices[i:i+chunk_size] for i in range(0, len(mol_indices), chunk_size)]
        
        worker_fn = partial(
            process_chunk_max_n_atoms,
            factory_datadir=self.factory.datadir,
            factory_mode='r',
            factory_swmr=True,
            factory_db_path=getattr(self.factory.mol_idx_mapper, 'db_path', None),
            remove_h=remove_h
        )
        
        print(f"Computing max_n_atoms using {num_workers} processes...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(worker_fn, chunks), total=len(chunks), desc="Computing max_n_atoms"))
        
        max_n_atoms = max(results)
        print(f"Found max_n_atoms = {max_n_atoms} (remove_h = {remove_h})")
        return max_n_atoms

    def compute_n_atoms_distribution(self, mol_indices: List[int], num_workers: Optional[int] = None, remove_h: bool = False) -> Dict[str, Any]:
        """Compute the distribution of number of atoms using multiprocessing."""
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, len(mol_indices))
        
        chunk_size = len(mol_indices) // num_workers + (1 if len(mol_indices) % num_workers else 0)
        chunks = [mol_indices[i:i+chunk_size] for i in range(0, len(mol_indices), chunk_size)]
        
        worker_fn = partial(
            process_chunk_n_atoms_distribution,
            factory_datadir=self.factory.datadir,
            factory_mode='r',
            factory_swmr=True,
            factory_db_path=getattr(self.factory.mol_idx_mapper, 'db_path', None),
            remove_h=remove_h
        )
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(worker_fn, chunks), total=len(chunks), desc="Computing n_atoms distribution", disable=True))
        
        # Flatten all results into a single list
        all_n_atoms = []
        for chunk_results in results:
            all_n_atoms.extend(chunk_results)
        
        if not all_n_atoms:
            print("No atom counts found.")
            return {
                'distribution': {},
                'statistics': {'min': 0, 'max': 0, 'mean': 0, 'std': 0, 'count': 0}
            }
        
        # Compute distribution and statistics
        distribution = dict(Counter(all_n_atoms))
        n_atoms_array = np.array(all_n_atoms)
        
        statistics = {
            'min': int(np.min(n_atoms_array)),
            'max': int(np.max(n_atoms_array)),
            'mean': float(np.mean(n_atoms_array)),
            'std': float(np.std(n_atoms_array)),
            'count': len(all_n_atoms),
            'median': float(np.median(n_atoms_array)),
            'percentile_25': float(np.percentile(n_atoms_array, 25)),
            'percentile_75': float(np.percentile(n_atoms_array, 75))
        }
        
        print("-" * 50)
        is_heavy_atom = "Heavy" if remove_h else "All"
        print(f"Distribution of # {is_heavy_atom} Atoms:")
        print(f"  # Data = {statistics['count']} molecules")
        print(f"  # {is_heavy_atom} Atoms in [{statistics['min']}, {statistics['max']}]")
        print(f"  Mean: {statistics['mean']:.2f}, Std: {statistics['std']:.2f}")
        # print(f"  25th percentile: {statistics['percentile_25']:.1f}")
        # print(f"  75th percentile: {statistics['percentile_75']:.1f}")
        print(f"  Distribution (top 10): {dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        return {
            'distribution': distribution,  # Dict[int, int] - n_atoms -> count
            'statistics': statistics,      # Dict with min, max, mean, std, etc.
            'raw_data': all_n_atoms       # List[int] - all n_atoms values
        }
    
    def compute_atom_types(self, mol_indices: List[int], num_workers: Optional[int] = None) -> List[str]:
        """Compute unique atom types using multiprocessing."""
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, len(mol_indices))
        
        chunk_size = len(mol_indices) // num_workers + (1 if len(mol_indices) % num_workers else 0)
        chunks = [mol_indices[i:i+chunk_size] for i in range(0, len(mol_indices), chunk_size)]
        
        worker_fn = partial(
            process_chunk_atom_types,
            factory_datadir=self.factory.datadir,
            factory_mode='r',
            factory_swmr=True,
            factory_db_path=getattr(self.factory.mol_idx_mapper, 'db_path', None)
        )
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(worker_fn, chunks), total=len(chunks), desc="Computing atom_types", disable=True))
        
        # Combine all atom types from all chunks and remove duplicates
        all_atom_types = set()
        for chunk_atom_types in results:
            all_atom_types.update(chunk_atom_types)
        
        unique_atom_types = sorted(list(all_atom_types))
        print("-" * 50)
        print(f"{len(unique_atom_types)} Atom Types: {unique_atom_types}")
        return unique_atom_types
    
    def compute_solvents_distribution(self, mol_indices: List[int], num_workers: Optional[int] = None, 
                                     solvent_type: Optional[str] = None) -> Dict[str, Any]:
        """Compute the distribution of solvents using multiprocessing."""
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        num_workers = min(num_workers, len(mol_indices))
        
        chunk_size = len(mol_indices) // num_workers + (1 if len(mol_indices) % num_workers else 0)
        chunks = [mol_indices[i:i+chunk_size] for i in range(0, len(mol_indices), chunk_size)]
        
        worker_fn = partial(
            process_chunk_solvents,
            factory_datadir=self.factory.datadir,
            factory_mode='r',
            factory_swmr=True,
            factory_db_path=getattr(self.factory.mol_idx_mapper, 'db_path', None),
            solvent_type=solvent_type
        )
        
        solvent_filter_msg = f" ({solvent_type})" if solvent_type else ""
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(worker_fn, chunks), total=len(chunks), desc="Computing solvents distribution",
                             disable=True))
        
        # Flatten all results into a single list (preserving duplicates for counting)
        all_solvents = []
        for chunk_solvents in results:
            all_solvents.extend(chunk_solvents)
        
        print("-" * 50)
        # Filter out empty strings
        filtered_solvents = [s for s in all_solvents if s and s.strip()]
        
        if not filtered_solvents:
            print(f"No solvents found{solvent_filter_msg}.")
            return {
                'distribution': {},
                'statistics': {'total_count': 0, 'unique_count': 0, 'most_common': []},
                'raw_data': []
            }
        
        # Compute distribution
        distribution = dict(Counter(filtered_solvents))
        
        # Compute statistics
        total_count = len(filtered_solvents)
        unique_count = len(distribution)
        most_common = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        
        statistics = {
            'total_count': total_count,
            'unique_count': unique_count,
            'most_common': most_common,  # Top 10 most common solvents
            'coverage_top5': sum(count for _, count in most_common[:5]) / total_count if total_count > 0 else 0,
            'coverage_top10': sum(count for _, count in most_common[:10]) / total_count if total_count > 0 else 0,
        }
        print(f"Solvents distribution{solvent_filter_msg}:")
        print(f"  # Total solvent entries: {statistics['total_count']}")
        print(f"  # Unique solvents: {statistics['unique_count']}")
        print(f"  Top 5 coverage: {statistics['coverage_top5']:.2%}")
        print(f"  Top 10 coverage: {statistics['coverage_top10']:.2%}")
        print(f"  Most common solvents: {dict(most_common[:10])}")
        
        return {
            'distribution': distribution,      # Dict[str, int] - solvent -> count
            'statistics': statistics,          # Dict with total_count, unique_count, etc.
            'raw_data': filtered_solvents     # List[str] - all solvent values
        }