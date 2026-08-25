# Protein Domain Boundary Prediction Pipeline

This repository contains the complete computational workflow for Seq2DomML's predicting protein domain boundaries from sequence data, including preprocessing, feature extraction, model training, and evolutionary algorithm driven feature selection.

## Directory structure

scripts.py/ 
- All scripts used in computational workflow including dependencies, preprocessing, feature extraction, model training, and evolutionary algorithm. Each script is identified in the order it must be run via the integer at the end of its name, beginning at 0, and finishing at 7. This excludes dependencies. 
	- scripts/dependencies.txt/ 
		- all dependencies required for installation. 
	- rcsb_fetch0.py/ 
		- Queries the RCSB PDB GraphQL API to download protein sequences for a list of target PDB IDs. It filters out non-protein entities and screens out membrane- or lipoprotein-related proteins based on keyword matches in titles and descriptions before writing out the valid sequences into data/preprocessing/sequences.csv (containing Entry ID and Sequence columns).
	- scripts/targ1.py/
		- Generates the established binary targets as ground truth labels(targ_binary) for each protein sequence from the ECOD dataset. It maps these known domain boundary ranges onto the sequences, trims boundary region lengths, enforces minimum linker gap lengths between adjacent domains, and outputs a binary mask where domain core regions are marked as 0 and linker/boundary regions are marked as 1, matching the length of the input sequence. It utilizes data/target_labeling/ids_from_ecod.txt to acquire such target label information. Output is data/preprocessing/targ_encoded.csv (containing Entry ID, Sequence, and targ_binary strings)
	- scripts/dedup2.py/
		- Deduplicates sequence records to prevent data redundancy and leakage across sequences that share over 90% global amino acid similarity in the first 50 residues. This is performed via pairwise global sequence alignments, and clusters sequences sharing over 90% identity across the first 50 residues in order to retain only one unique representative sequence.
	- scripts/encode_features3.py/
		- Transforms the deduplicated protein sequences into numerical feature matrices suitable for the machine learning pipeline. It constructs positional encodings (relative position, distance to terminus, sinusoidal), amino acid one-hot vectors, physical/chemical index properties, and pairwise interaction matrices, all present in features/ folder. It normalizes continuous values across the dataset, saves scalers, and outputs individual compressed NumPy arrays for each sequence.
	- scripts/split4.py/
		- Performs a random, deterministic 65/35 dataset split on the encoded .npz sequence files to isolate sequences designated into the train set, and sequences designated as the validation and test set, which will later be further split up for the lstm and evolutionary algorithm. It specifically outputs target IDs, outputs ID manifests for future use, copies split .npz files into separate directories, and generates a indexed tab separated feature name lookup file, in order to ensure feature names are retainable after feature selection is performed in a later step.
	- scripts/prune5.py/
		- Filters continuous feature dimensions based on point-biserial feature-target correlations computed across the training set. It ranks all features by correlation magnitude, generates a summary correlation bar plot, and creates multiple pruned sub-datasets across multiple predefined target retention fractions along with their feature manifests, once again useful f for feature name retention in later step of the evolutionary algorithm.
	- scripts/lstm6.py/
		- Trains and evaluates a Bidirectional LSTM model with the custom residue type weighting and class balancing used in this workflow. It monitors validation macro-F1/loss for early stopping, sweeps prediction thresholds, applies post-processing domain smoothing rules, and outputs quantitative performance metrics alongside diagnostic plots, and saves the model. In order to run the full set of correlation pruning trials, this script must be re-run multiple times for each saved output from the prior script. Thereby, in order to run this script on the variant of the feature set with the 25% most correlated features, the following 
	TRAIN_CACHE_DIR = Path("data/pruned_featuresets/1.0/train")
	HOLDOUT_CACHE_DIR = Path("data/pruned_featuresets/1.0/test_and_val") must be changed to the following:
	TRAIN_CACHE_DIR = Path("data/pruned_featuresets/0.25/train")
	HOLDOUT_CACHE_DIR = Path("data/pruned_featuresets/0.25/test_and_val")
In addition, output path should also be adjusted accordingly.
	- scripts/run_evolution.py/
		- This script executes the Evolutionary Algorithm driven Feature Selection process paired with the Bidirectional LSTM (BiLSTM) architecture on the selected feature subset from correlation pruning trials. Its primary purpose is to find optimal subsets of sequence features within the selected correlation pruned subset that maximizes three contradictory objectives; the non-domain (label 1) precision, the non domain (label 1) recall, and the macro F1 (the mean F1 scores of the non domain (label 0) and within domain classes (label 1). The evolutionary algorithm is performed on a subset of the training and validation data for computational efficiency, and is completed with a final retraining phase for the final population on the full size training, validation, and test sets.


data/
- All data used for primary input into the pipeline.
	- pdb_entry_ids/entry_ids.txt/
		- The list of PDB Entry IDs downloaded from RCSB PDB, matching the kingdom of bacterial sequences. Used as input for rcsb_fetch0.py script, in order to filter out non-desirable sequences.
	- target_labeling/ids_from_ecod.txt/
		- The opening string of each line (e.g., 19hcA, 1a04B) specifies the 4 character PDB code alongside the corresponding PDB Entry ID alongside the Chain identifier of the sequence in this archive. Following the semicolon, the file defines domain assignments and their exact residue "within boundary" ranges (e.g., 1 [A:1-168]) which is allocated as our within domain class label. 


features/
- All features stored for amino acid feature extraction, extracted within the scripts/encode_features3.py. 
	- interaction_matrices/
		- Contains the interaction matrix CSV files used to generate the interaction matrix derived feature group, downloaded from aaindex.org. This folder contains 90 interaction matrix files, with each file storing a 20 by 20 amino acid matrix. Each matrix defines numerical scores for pairwise relationships between the 20 standard amino acids. During feature extraction, each matrix contributes 20 residue-level features, one for each amino acid column in the matrix. Across the 90 interaction matrices, this produces 1,800 total interaction matrix derived features in the full feature set. These features are named using the amino acid of interest followed by the matrix identifier, allowing each feature to be traced back to the corresponding CSV file in this folder.
	- aa_indices/indices.txt/
		- Contains the amino acid index resources used to generate the residue-level physicochemical feature group. These files define the lookup tables used to convert each amino acid residue into numerical values representing biochemical or physicochemical properties.
	- aa_indices/lookup_tables.txt/
		- Contains the ordered list of amino acid indices feature names for reference in the feature extraction workflow. This file serves as a manifest for the available amino acid index features and links each lookup table name to its corresponding dictionary in the aa_indices/indices.txt.
	- bin_feats/bin_feats.txt
		- Methodology described for extracting amino acid binary features. Note; this method is already hardcoded into the scripts/encode_features3.py script. 
	- positional_feats/pos_feats.txt
		- Methodology described for extracting the three positional features for each amino acid. Note; this method is already hardcoded into the scripts/encode_features3.py script.