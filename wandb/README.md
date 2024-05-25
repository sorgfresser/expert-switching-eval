# Starting the sweep

To start the sweep, run the following command:

```bash
wandb sweep --project ehrsql_sweep config.yml
```

Note that `Sweep ID` will be printed to the console. This ID is used to start the agents.

# Starting agents

To start an agent, run the following command:

```bash
wandb agent <SWEEP_ID>
```

If you want to spin up multiple agents on the same machine using CUDA, be sure to set `CUDA_VISIBLE_DEVICES` to the desired GPU indices:

```bash
CUDA_VISIBLE_DEVICES=<INDICES_OF_AGENT> wandb agent <SWEEP_ID>
```
