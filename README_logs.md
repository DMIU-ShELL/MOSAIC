# Understanding the MOSAIC log directory
Below is a sample log directory structure of a MOSAIC experiment (one seed run).

```
MetaCTgraph-shell-dist-ct28_a0_d1-seed-9157/agent_0/250512-134235/train-log-250512-134235.txt
MetaCTgraph-shell-dist-ct28_a0_d1-seed-9157/
├── agent_0/
│   └── 220902-132612/
│       ├── .....
│       ├── Detect_Component_Generated_Embeddings/
│       ├── betas.csv
│       ├── exchanges.csv
│       ├── masks.csv
│       ├── metadata.csv
│       ├── requests.csv
│       ├── parameters.txt
│       ├── shell_config.json
│       ├── train-log-250512-134235.txt
│       ├── eval_metrics_agent_0.csv
│       ├── eval_metrics_agent_0.npy
│       ├── .....
│       └── 
└── agent_1/
    └── 220902-132617/
        ├── .....
        ├── Detect_Component_Generated_Embeddings/
        ├── eval_metrics_agent_1.csv
        ├── eval_metrics_agent_1.npy
        ├── .....
        └── 
```

WIP