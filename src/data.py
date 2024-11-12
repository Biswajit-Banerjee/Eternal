import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tabulate import tabulate
import json
import pickle
import os
from pathlib import Path

class RNADataset(Dataset):
    def __init__(self, sequences, structures):
        self.sequences = sequences
        self.structures = structures

        # Nucleotide to index mapping
        self.nt_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
        # Structure to index mapping
        self.struct_to_idx = {
            ".": 0,  # unpaired
            "(": 1,  # opening bracket
            ")": 2,  # closing bracket
            "[": 1,  # alternate opening bracket
            "]": 2,  # alternate closing bracket
            "{": 1,  # alternate opening bracket
            "}": 2,  # alternate closing bracket
            "<": 1,  # alternate opening bracket
            ">": 2   # alternate closing bracket
        }
        
        for i in range(26):
            self.struct_to_idx[chr(ord("A")+i)] = 1 # Alternate opening bracket
            self.struct_to_idx[chr(ord("a")+i)] = 2 # Alternate close bracket
        

    def __len__(self):
        return len(self.sequences)

    def info(self):
        """
        Provides comprehensive statistics about the RNA sequences and structures
        in the dataset using pretty table formatting.
        """
        # Sequence length statistics
        seq_lens = [len(seq) for seq in self.sequences]
        total_nts = sum(seq_lens)

        # Basic Statistics Table
        basic_stats = [
            ["Total Sequences", len(self.sequences)],
            ["Total Nucleotides", total_nts],
            ["Average Length", f"{total_nts/len(self.sequences):.2f}"],
            ["Min Length", min(seq_lens)],
            ["Max Length", max(seq_lens)],
            ["Std Dev Length", f"{np.std(seq_lens):.2f}"],
        ]

        print("\n=== Basic Dataset Statistics ===")
        print(
            tabulate(
                basic_stats,
                headers=["Metric", "Value"],
                tablefmt="pretty",
                numalign="right",
            )
        )

        # Nucleotide Frequencies Table
        all_nts = "".join(self.sequences)
        nt_counts = Counter(all_nts)

        nt_stats = []
        for nt in ["A", "U", "G", "C"]:
            count = nt_counts.get(nt, 0)
            freq = (count / total_nts) * 100
            nt_stats.append([nt, count, f"{freq:.2f}%"])

        gc_content = ((nt_counts.get("G", 0) + nt_counts.get("C", 0)) / total_nts) * 100
        nt_stats.append(["GC Content", "-", f"{gc_content:.2f}%"])

        print("\n=== Nucleotide Composition ===")
        print(
            tabulate(
                nt_stats,
                headers=["Nucleotide", "Count", "Frequency"],
                tablefmt="pretty",
                numalign="right",
            )
        )

        # Structure Statistics Table
        all_structs = "".join(self.structures)
        struct_counts = Counter(all_structs)

        struct_stats = []
        for struct in [".", "(", ")"]:
            count = struct_counts.get(struct, 0)
            freq = (count / total_nts) * 100
            struct_stats.append([struct, count, f"{freq:.2f}%"])

        print("\n=== Structure Composition ===")
        print(
            tabulate(
                struct_stats,
                headers=["Element", "Count", "Frequency"],
                tablefmt="pretty",
                numalign="right",
            )
        )

        # Base Pair Statistics
        total_pairs = struct_counts.get("(", 0)
        bp_stats = [
            ["Total Base Pairs", total_pairs],
            ["Avg Pairs per Sequence", f"{total_pairs/len(self.sequences):.2f}"],
            ["Base Pair Density", f"{(total_pairs*2/total_nts)*100:.2f}%"],
        ]

        print("\n=== Base Pair Statistics ===")
        print(
            tabulate(
                bp_stats,
                headers=["Metric", "Value"],
                tablefmt="pretty",
                numalign="right",
            )
        )

        # Common Dinucleotides Table
        dinucleotides = [all_nts[i : i + 2] for i in range(len(all_nts) - 1)]
        di_counts = Counter(dinucleotides).most_common(5)
        di_stats = []
        for di, count in di_counts:
            freq = (count / (total_nts - 1)) * 100
            di_stats.append([di, count, f"{freq:.2f}%"])

        print("\n=== Most Common Dinucleotides ===")
        print(
            tabulate(
                di_stats,
                headers=["Dinucleotide", "Count", "Frequency"],
                tablefmt="pretty",
                numalign="right",
            )
        )

        # Structure Context Analysis
        contexts = {}
        for seq, struct in zip(self.sequences, self.structures):
            for nt, st in zip(seq, struct):
                context = (nt, st)
                contexts[context] = contexts.get(context, 0) + 1

        context_stats = []
        for nt in ["A", "U", "G", "C"]:
            unpaired = contexts.get((nt, "."), 0)
            paired = contexts.get((nt, "("), 0) + contexts.get((nt, ")"), 0)
            if unpaired + paired > 0:
                paired_pct = (paired / (unpaired + paired)) * 100
                unpaired_pct = 100 - paired_pct
                total = paired + unpaired
                context_stats.append(
                    [nt, total, f"{paired_pct:.1f}%", f"{unpaired_pct:.1f}%"]
                )

        print("\n=== Nucleotide Pairing Statistics ===")
        print(
            tabulate(
                context_stats,
                headers=["Nucleotide", "Total", "Paired", "Unpaired"],
                tablefmt="pretty",
                numalign="right",
            )
        )

    def encode_sequence(self, seq):
        # Convert sequence to indices
        seq_idx = [self.nt_to_idx.get(nt, self.nt_to_idx["N"]) for nt in seq]
        # One-hot encode
        one_hot = np.zeros((len(seq), len(self.nt_to_idx)))
        one_hot[np.arange(len(seq)), seq_idx] = 1
        return torch.FloatTensor(one_hot)

    def encode_structure(self, struct):
        # Convert structure to indices
        struct_idx = [self.struct_to_idx[s] for s in struct]
        return torch.LongTensor(struct_idx)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        struct = self.structures[idx]

        # Get encodings
        seq_encoding = self.encode_sequence(seq)
        struct_encoding = self.encode_structure(struct)

        # Create base pair matrix (for advanced training)
        bp_matrix = self.create_bp_matrix(struct)

        return {
            "sequence": seq_encoding,
            "structure": struct_encoding,
            "bp_matrix": bp_matrix,
            "length": len(seq),
        }

    def create_bp_matrix(self, struct):
        n = len(struct)
        matrix = np.zeros((n, n))
        stack = []

        for i, s in enumerate(struct):
            if s == "(":
                stack.append(i)
            elif s == ")" and stack:
                j = stack.pop()
                matrix[i][j] = matrix[j][i] = 1

        return torch.FloatTensor(matrix)
    
    def save(self, filepath, format='json'):
        """
        Save the dataset to a file.
        
        Args:
            filepath: Path to save the file
            format: 'json', 'pickle', or 'txt'
        """
        data = {
            'sequences': self.sequences,
            'structures': self.structures,
            'nt_to_idx': self.nt_to_idx,
            'struct_to_idx': self.struct_to_idx
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
    
        open_func = open
        mode = 'w' if format == 'txt' else 'wb'
        
        with open_func(filepath, mode) as f:
            if format == 'json':
            
                json.dump(data, f, indent=2)
            elif format == 'pickle':
                pickle.dump(data, f)
            elif format == 'txt':
                # Custom text format: sequence\tstructure
                for seq, struct in zip(self.sequences, self.structures):
                    f.write(f"{seq}\t{struct}\n")
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        print(f"Dataset saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load the dataset from a file.
        
        Args:
            filepath: Path to the saved dataset
        """
        is_compressed = filepath.endswith('.gz')
        open_func = open
        mode = 'rt' if filepath.endswith(('.txt.gz', '.txt')) else 'rb'
        
        with open_func(filepath, mode) as f:
            if filepath.endswith(('.json.gz', '.json')):
                if is_compressed:
                    data = json.loads(f.read())
                else:
                    data = json.load(f)
            elif filepath.endswith(('.pkl.gz', '.pkl')):
                data = pickle.load(f)
            elif filepath.endswith(('.txt.gz', '.txt')):
                # Parse custom text format
                sequences = []
                structures = []
                for line in f:
                    seq, struct = line.strip().split('\t')
                    sequences.append(seq)
                    structures.append(struct)
                data = {
                    'sequences': sequences,
                    'structures': structures
                }
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        
        dataset = cls(data['sequences'], data['structures'])
        if 'nt_to_idx' in data:
            dataset.nt_to_idx = data['nt_to_idx']
        if 'struct_to_idx' in data:
            dataset.struct_to_idx = data['struct_to_idx']
            
        return dataset
    
    @staticmethod
    def _pairs_to_dot_bracket(pairs):
        """Convert base pair indices to dot-bracket notation"""
        length = len(pairs)
        structure = ['.' for _ in range(length)]
        
        for i, pair in enumerate(pairs, 1):
            if pair > i:  # Opening bracket
                structure[i-1] = '('
                structure[pair-1] = ')'
                
        return ''.join(structure)
