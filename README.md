# Codes for LREC-COLING 2024 Submission 3115: "Dual Complex Number Knowledge Graph Embeddings"

This is the official implementation of DCNE.

## Requirements

To install requirements:

```
conda env create -f requirements.yaml
```

After installing the requirements, please run this command to activate.

```
conda activate dcne
```

## Reproducing the best results

The datasets for link prediction can be found in "LP/data" directory, including WN18, WN18RR, FB15k, FB15k-237 and Countries. For path query answering datasets WordNet and Freebase, please run the following commands to download.

```
cd PQA
sh prepare_datasets.sh
```

### Link Prediction and Countries dataset

#### Main results

Please run this command first:

```
cd LP
```

To reproduce the results on **WN18RR**, run the following command:

```
bash run.sh train DCNE wn18rr 0 0 1024 512 1000 6.0 1.0 0.00005 80000 8 -de -r 0.1 --wn18rr
```

To reproduce the results on **WN18**, run the following command:

```
bash run.sh train DCNE wn18 0 0 512 256 1000 12.0 0.5 0.0001 80000 8 -de -r 0.1 --wn18
```

To reproduce the results on **FB15k-237**, run the following command:

```
bash run.sh train DCNE FB15k-237 0 0 1024 128 1000 9.0 1.0 0.0001 100000 16 -de --FB15k_237
```

To reproduce the results on **FB15k**, run the following command:

```
bash run.sh train DCNE FB15k 0 0 1024 256 1000 24.0 0.5 0.0002 150000 16 -de --FB15k
```

To reproduce the results on **Countries S1**, run the following command:

```
bash run.sh train DCNE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
```

To reproduce the results on **Countries S2**, run the following command:

```
bash run.sh train DCNE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
```

To reproduce the results on **Countries S3**, run the following command:

```
bash run.sh train DCNE countries_S3 0 0 512 64 1000 0.1 0.2 0.000002 40000 8 -de --countries
```

#### Ablation study on WN18RR and FB15k-237

To reproduce the ablation study results on WN18RR, run the following command:

```
bash run_ablation_wn18rr.sh
```

To reproduce the ablation study results on FB15k-237, run the following command:

```
bash run_ablation_FB15k-237.sh
```

### Path Query Answering

Please run this command first (if you do not download the datasets, please download Freebase and WordNet datasets according to the above instructions before running the code of path query answering):

```
cd PQA
```

To reproduce the results on **Freebase**, run the following command after preparing the datasets:

```
bash runs.sh comp train DCNE pathqueryFB 0 0 1024 512 1000 12.0 1.0 0.00002 80000 0 2
```

To reproduce the results on **WordNet**, run the following command after preparing the datasets:

```
bash runs.sh comp train DCNE pathqueryWN 0 0 1024 256 1000 6.0 1.0 0.00002 80000 0.1 2
```

## License

The code of DCNE is licensed under the MIT license.

## Acknowledgement

We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding), [Rotate3D](https://github.com/gao-xiao-bai/Rotate3D-Representing-Relations-as-Rotations-in-Three-Dimensional-Space-for-KG-Embedding). Thanks for their contributions.