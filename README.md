# traNNsformers
# Structured clustering for memristive crossbar based neuromorphic architectures

## Objective: Reduce the number of crossbars getting mapped to unclustered sunapses.
Approach 2a) Decrease cluster_base_quality_min to form clusteres with lower utilization too.
  Advantage: Existing framework will fill up such crossbars and the new synapses can add to accuracy.
  
## Put the cluster_prune technique in the training framework the same way as discrete synapse pruning is put.
  >> Run cluster_prune
  >> If (accuracy degrades)
        Recover accuracy in subsequent training epochs.
     Else
        Increase cluster_prune_threshold.
