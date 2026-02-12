# Behavior Tree Generator Prototype

A Python tool that generates random behavior tree diagrams. The trees are visually structured, exportable as images, and configurable with multiple parameters.

### What This Tool Does

This tool creates behavior tree graphs that represent sequences and decisions an AI agent might take. The focus is on generating structurally valid and visually appealing trees, not functional AI for games.

Each tree is exported as:

- A DOT file (Graphviz format)
- A PNG image for easy viewing

You can customize how many trees are generated, their depth, visual style, and probability of control vs leaf nodes.

## Requirements

- Python 3.9+
- Graphviz installed and available in PATH (for generating PNGs)

## How to Run

### 1. Install Requirements

Make sure you have:

- Python 3.9+
- Graphviz installed and added to your system PATH

To verify Graphviz is installed:

```
dot -V
```

If installed correctly, it should print the Graphviz version.

### 2. Prepare Required Files

By default, the program looks for these files in the project directory:

- `grammar.json`
- `blackboard.json`
- `behaviors.json`

You can use these default filenames or specify custom ones with CLI flags.

### 3. Run the Generator (Default Settings)

```
python behavior_tree.py
```

This will:

- Load grammar.json, blackboard.json, and behaviors.json
- Generate trees using default parameters
- Save .dot files to output/dot/
- Save .png images to output/png/

### 4. Customize Parameters via CLI

You can override generation settings using flags:

```
python behavior_tree.py \
  --max-depth 8 \
  --control-prob 0.8 \
  --layout circo \
  --num 10
```

| Flag                  | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| `--grammar`           | Path to grammar JSON file                                                 |
| `--blackboard`        | Path to blackboard JSON file                                              |
| `--behaviors`         | Path to behaviors JSON file                                               |
| `--max-depth`         | Maximum tree depth                                                        |
| `--control-prob`      | Probability of control nodes (0–1)                                        |
| `--min-control-depth` | Ensures the top N levels have at least one control node                   |
| `--layout`            | Graphviz layout engine (`dot`, `circo`, `fdp`, `neato`, `osage`, `twopi`) |
| `--num`               | Number of trees to generate                                               |
| `--transparent`       | Transparent PNG background                                                |

Example using custom files:

```
python behavior_tree.py \
  --grammar my_grammar.json \
  --blackboard my_blackboard.json \
  --behaviors my_behaviors.json
```

### Known Limitations

These trees are not functional AI, they won’t control a game agent.

Leaves are randomly sampled, the behavior tree may not make logical sense.

Very large trees may freeze.

## Citation / AI Use

Generative AI was used to assist in code writing for the behavior tree generator and documentation.