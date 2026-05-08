# PPO BipedalWalker

Deep reinforcement learning notebook for training an actor-critic agent on `BipedalWalker-v3`.

The project implements a policy-gradient workflow with rollout collection, advantage estimation, training updates, checkpointing, evaluation, and model export. The notebook was adapted to run in a local environment instead of a Colab-only setup.

## Project Goals

- Train a deep reinforcement learning agent for continuous control
- Use a policy-gradient / PPO-style learning pipeline
- Collect environment rollouts and compute advantages
- Evaluate the trained agent visually and numerically
- Save checkpoints and export the final model

## Main Components

- Environment setup using `gymnasium`
- Actor-Critic neural networks in PyTorch
- Rollout buffer and episode transition handling
- Training loop with clipped policy updates
- Observation and return normalization
- Checkpoint saving and loading
- Video recording for agent evaluation
- ONNX export for inference portability

## Notebook Workflow

1. Initialize imports and runtime settings
2. Create and wrap the environment
3. Test the environment with a random agent
4. Define transition storage and rollout buffers
5. Build actor and critic networks
6. Define the agent and training utilities
7. Train the agent with rollout-based updates
8. Save the best checkpoint
9. Reload the trained model
10. Run evaluation episodes and record videos
11. Export the model to ONNX

## Local Execution Notes

The notebook was modified to run locally by adjusting environment detection, display handling, device selection, and file paths. It no longer depends on a Colab-only runtime.

## Requirements

Typical dependencies include:

- `torch`
- `gymnasium`
- `numpy`
- `matplotlib`
- `seaborn`
- `onnx`
- `onnx2pytorch`
- `tqdm`
- `pyvirtualdisplay`

## Outputs

The notebook produces:

- Training logs
- Saved checkpoints
- Evaluation videos
- ONNX model export
- Performance plots

## Usage

Run the notebook cell by cell after installing the required dependencies and ensuring the environment is available.
