"""
Behavior Tree Prototype (Single File)

Requirements:
- Python 3.9+
- Graphviz installed (for visualization)

Usage:
    python behavior_tree.py <grammar.json> <blackboard.json> <behaviors.json>
    dot -Tpng bt.dot -o bt.png
"""


from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Optional
import sys
import json
import random
import uuid

# =========================
# Generator Parameters
# =========================

MAX_DEPTH = 5
CONTROL_NODE_PROB = 0.5
GRAPHVIZ_LAYOUT = "twopi" # dot, circo, fdp, neato, osage, patchwork, twopi
NUM_TREES = 3

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_inputs():
    if len(sys.argv) < 4:
        print("Usage: python behavior_tree.py <grammar.json> <blackboard.json> <behaviors.json>")
        sys.exit(1)

    grammar = load_json(sys.argv[1])
    blackboard = load_json(sys.argv[2])
    behaviors = load_json(sys.argv[3])

    print("Loaded grammar keys:", grammar.keys())
    print("Loaded blackboard:", blackboard)
    print("Loaded behaviors:", behaviors.keys())

    return grammar, blackboard, behaviors

def evaluate_condition(cond, blackboard):
    key = cond["key"]
    op = cond["op"]
    value = cond["value"]
    bb_value = blackboard.get(key)

    if op == "==": return bb_value == value
    if op == "!=": return bb_value != value
    if op == ">":  return bb_value > value
    if op == "<":  return bb_value < value
    if op == ">=": return bb_value >= value
    if op == "<=": return bb_value <= value

    raise ValueError(f"Unknown operator {op}")

def execute_action(action, blackboard):
    if action["type"] == "noop":
        return Status.SUCCESS

    if action["type"] == "modify":
        key = action["key"]
        op = action["op"]
        value = action["value"]

        if op == "+=":
            blackboard[key] = blackboard.get(key, 0) + value
        elif op == "-=":
            blackboard[key] = blackboard.get(key, 0) - value
        else:
            raise ValueError(f"Unknown modify op {op}")

        return Status.SUCCESS

    return Status.FAILURE

# =========================
# Execution Status
# =========================

class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


# =========================
# Base Node
# =========================

class BTNode(ABC):
    def __init__(self, name: str):
        self.id = str(uuid.uuid4()).replace("-", "")
        self.name = name
        self.parent: Optional["BTNode"] = None
        self.children: List["BTNode"] = []
        self.last_status: Optional[Status] = None

    def add_child(self, child: "BTNode"):
        child.parent = self
        self.children.append(child)

    def label(self):
        status = self.last_status.name if self.last_status else "NONE"
        return f"{self.__class__.__name__}\\n{self.name}\\n[{status}]"

    @abstractmethod
    def tick(self, blackboard) -> Status:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


# =========================
# Control Nodes
# =========================

class Sequence(BTNode):
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.SUCCESS:
                self.last_status = status
                return status
        self.last_status = Status.SUCCESS
        return Status.SUCCESS


class Selector(BTNode):
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != Status.FAILURE:
                self.last_status = status
                return status
        self.last_status = Status.FAILURE
        return Status.FAILURE


# =========================
# Leaf Nodes
# =========================

class Condition(BTNode):
    def __init__(self, name, condition_data):
        super().__init__(name)
        self.condition = condition_data

    def tick(self, blackboard):
        return Status.SUCCESS if evaluate_condition(self.condition, blackboard) else Status.FAILURE

class Action(BTNode):
    def __init__(self, name, action_data):
        super().__init__(name)
        self.action = action_data

    def tick(self, blackboard):
        return execute_action(self.action, blackboard)

# =========================
# Grammar-Based Generation
# =========================

# GRAMMAR = {
#     "NODE": [("CONTROL", 0.8), ("LEAF", 0.2)],
#     "CONTROL": [("SEQUENCE", 0.5), ("SELECTOR", 0.5)],
#     "SEQUENCE": [("Sequence", ["NODE"])],  # children added randomly
#     "SELECTOR": [("Selector", ["NODE"])],
#     "LEAF": [("CONDITION", 0.5), ("ACTION", 0.5)],
# }


def weighted_choice(options):
    symbols, weights = zip(*options)
    return random.choices(symbols, weights=weights)[0]

def expand(symbol, depth, max_depth):
    if depth >= max_depth:
        return (weighted_choice(list(GRAMMAR["leaf"].items())),)

    if symbol == "NODE":
        control_prob = CONTROL_NODE_PROB
        leaf_prob = 1.0 - control_prob

        choice = weighted_choice([
            ("CONTROL", control_prob),
            ("LEAF", leaf_prob)
        ])
        return expand(choice, depth + 1, max_depth)

    if symbol == "CONTROL":
        choice = weighted_choice(list(GRAMMAR["control"].items()))
        return expand(choice, depth + 1, max_depth)

    if symbol in ("SEQUENCE", "SELECTOR"):
        num_children = random.randint(
            GRAMMAR["children"]["min"],
            GRAMMAR["children"]["max"]
        )
        return (
            symbol,
            [expand("NODE", depth + 1, max_depth) for _ in range(num_children)]
        )

    if symbol == "LEAF":
        return expand(weighted_choice(list(GRAMMAR["leaf"].items())), depth + 1, max_depth)

    return (symbol,)

# =========================
# Behavior Tree Container
# =========================

class BehaviorTree:
    def __init__(self, root: BTNode):
        self.root = root

    def tick(self, blackboard):
        return self.root.tick(blackboard)


# =========================
# Visualization (Graphviz)
# =========================

class BTVisualizer:
    def __init__(self, tree: BehaviorTree):
        self.tree = tree
        self.lines = []

    def to_dot(self) -> str:
        self.lines.append("digraph BehaviorTree {")
        self.lines.append("  rankdir=TB;")
        self.lines.append("  node [fontname=Helvetica];")

        self._visit(self.tree.root)

        self.lines.append("}")
        return "\n".join(self.lines)

    def _visit(self, node: BTNode):
        style = self._node_style(node)
        self.lines.append(
            f'  "{node.id}" [label="{node.label()}" {style}];'
        )

        for child in node.children:
            self.lines.append(f'  "{node.id}" -> "{child.id}";')
            self._visit(child)


    def _node_style(self, node: BTNode) -> str:
        if isinstance(node, Sequence):
            return 'shape=box style="filled,rounded" fillcolor=lightblue'
        if isinstance(node, Selector):
            return 'shape=box style="filled,rounded" fillcolor=khaki'
        if isinstance(node, Condition):
            return 'shape=ellipse style="filled" fillcolor=palegreen'
        if isinstance(node, Action):
            return 'shape=ellipse style="filled" fillcolor=lightcoral'
        return 'shape=box'


def save_dot(tree: BehaviorTree, filename="bt.dot"):
    dot = BTVisualizer(tree).to_dot()
    with open(filename, "w") as f:
        f.write(dot)
    print(f"[OK] Behavior tree saved to {filename}")

# =========================
# Leaf Pools (Game Logic)
# =========================

CONDITIONS = [
    ("Enemy Visible?", lambda bb: bb["enemy_visible"]),
    ("Has Ammo?", lambda bb: bb["ammo"] > 0),
]

ACTIONS = []  # filled in main

def build_bt(ast):
    node_type = ast[0]
    
    if node_type == "SEQUENCE":
        node = Sequence("Sequence")
        for child in ast[1]:
            node.add_child(build_bt(child))
        return node

    if node_type == "SELECTOR":
        node = Selector("Selector")
        for child in ast[1]:
            node.add_child(build_bt(child))
        return node

    if node_type == "CONDITION":
        cond = random.choice(BEHAVIORS["conditions"])
        return Condition(cond["id"], cond)

    if node_type == "ACTION":
        act = random.choice(BEHAVIORS["actions"])
        return Action(act["id"], act)

    raise ValueError(f"Unknown AST node: {ast}")


def generate_random_tree(max_depth=5):
    root_symbol = random.choice(GRAMMAR["root"])
    ast = expand(root_symbol, 0, max_depth)
    root = build_bt(ast)
    return BehaviorTree(root)


# =========================
# Example / Test Harness
# =========================

if __name__ == "__main__":
    # Blackboard (game state)
    GRAMMAR, BLACKBOARD, BEHAVIORS = load_inputs()
    blackboard = dict(BLACKBOARD)

    # Conditions
    def enemy_visible(bb):
        return bb["enemy_visible"]

    def has_ammo(bb):
        return bb["ammo"] > 0

    # Actions
    def shoot(bb):
        bb["ammo"] -= 1
        print("Action: Shoot")
        return Status.SUCCESS

    def patrol(bb):
        print("Action: Patrol")
        return Status.SUCCESS

    ACTIONS.extend([
        ("Shoot", shoot),
        ("Patrol", patrol),
    ])

    # Build tree
    tree = generate_random_tree(max_depth=MAX_DEPTH)

    # Tick tree
    print("Tick result:", tree.tick(blackboard))

    # Visualize
    save_dot(tree)

import os
import subprocess

# =========================
# Batch Generation / Image Export (Organized Folders)
# =========================

# Generator parameters
# NUM_TREES = 5                # How many trees to generate
# GRAPHVIZ_LAYOUT = "dot"      # # dot, circo, fdp, neato, osage, patchwork, twopi
OUTPUT_FOLDER = "output"     # base output folder

# Create organized subfolders
DOT_FOLDER = os.path.join(OUTPUT_FOLDER, "dot")
PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "png")
os.makedirs(DOT_FOLDER, exist_ok=True)
os.makedirs(PNG_FOLDER, exist_ok=True)

for i in range(NUM_TREES):
    # Generate a new tree
    tree = generate_random_tree(max_depth=MAX_DEPTH)

    # Tick the tree (optional)
    result = tree.tick(blackboard)
    print(f"Tree {i+1} tick result:", result)

    # Save DOT file
    dot_path = os.path.join(DOT_FOLDER, f"bt_{i+1}.dot")
    save_dot(tree, filename=dot_path)

    # Save PNG via Graphviz
    png_path = os.path.join(PNG_FOLDER, f"bt_{i+1}.png")
    try:
        subprocess.run(
            [GRAPHVIZ_LAYOUT, "-Tpng", dot_path, "-o", png_path],
            check=True
        )
        print(f"[OK] Saved image {png_path}")
    except Exception as e:
        print(f"[ERROR] Could not generate image: {e}")
