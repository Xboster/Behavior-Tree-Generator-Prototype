import os
import sys
import json
import random
import uuid
import subprocess
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Optional
import argparse

# =========================
# CLI Arguments
# =========================
parser = argparse.ArgumentParser(description="Behavior Tree Generator")
parser.add_argument("--grammar", type=str, default="grammar.json")
parser.add_argument("--blackboard", type=str, default="blackboard.json")
parser.add_argument("--behaviors", type=str, default="behaviors.json")
parser.add_argument("--max-depth", type=int, default=5)
parser.add_argument("--control-prob", type=float, default=0.5)
parser.add_argument("--min-control-depth", type=int, default=1)
parser.add_argument("--num", type=int, default=3)
parser.add_argument("--layout", type=str, default="dot")
parser.add_argument("--transparent", action="store_true")
args = parser.parse_args()

# =========================
# Load JSON
# =========================
def load_json(path):
    with open(path) as f:
        return json.load(f)

GRAMMAR = load_json(args.grammar)
BLACKBOARD = load_json(args.blackboard)
BEHAVIORS = load_json(args.behaviors)

# =========================
# Status and Base Node
# =========================
class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()

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
        key = self.condition["key"]
        op = self.condition["op"]
        value = self.condition["value"]
        bb_value = blackboard.get(key)
        if op == "==": return Status.SUCCESS if bb_value == value else Status.FAILURE
        if op == "!=": return Status.SUCCESS if bb_value != value else Status.FAILURE
        if op == ">":  return Status.SUCCESS if bb_value > value else Status.FAILURE
        if op == "<":  return Status.SUCCESS if bb_value < value else Status.FAILURE
        if op == ">=": return Status.SUCCESS if bb_value >= value else Status.FAILURE
        if op == "<=": return Status.SUCCESS if bb_value <= value else Status.FAILURE
        raise ValueError(f"Unknown operator {op}")

class Action(BTNode):
    def __init__(self, name, action_data):
        super().__init__(name)
        self.action = action_data
    def tick(self, blackboard):
        if self.action["type"] == "noop":
            return Status.SUCCESS
        if self.action["type"] == "modify":
            key = self.action["key"]
            op = self.action["op"]
            value = self.action["value"]
            if op == "+=": blackboard[key] = blackboard.get(key,0)+value
            elif op == "-=": blackboard[key] = blackboard.get(key,0)-value
            return Status.SUCCESS
        return Status.FAILURE

# =========================
# Behavior Tree Builder
# =========================
def weighted_choice(options):
    symbols, weights = zip(*options)
    return random.choices(symbols, weights=weights)[0]

def expand(symbol, depth, max_depth):
    if depth >= max_depth:
        return (weighted_choice(list(GRAMMAR["leaf"].items())),)
    if symbol == "NODE":
        if depth < args.min_control_depth:
            choice = "CONTROL"
        else:
            leaf_prob = 1.0 - args.control_prob
            choice = weighted_choice([
                ("CONTROL", args.control_prob),
                ("LEAF", leaf_prob)
            ])
        return expand(choice, depth + 1, max_depth)
    if symbol == "CONTROL":
        choice = weighted_choice(list(GRAMMAR["control"].items()))
        return expand(choice, depth + 1, max_depth)
    if symbol in ("SEQUENCE","SELECTOR"):
        num_children = random.randint(
            GRAMMAR["children"]["min"], GRAMMAR["children"]["max"]
        )
        return (symbol, [expand("NODE", depth+1, max_depth) for _ in range(num_children)])
    if symbol == "LEAF":
        return expand(weighted_choice(list(GRAMMAR["leaf"].items())), depth+1, max_depth)
    return (symbol,)

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

def generate_random_tree(max_depth):
    root_symbol = random.choice(GRAMMAR["root"])
    ast = expand(root_symbol, 0, max_depth)
    return build_bt(ast)

# =========================
# Visualization
# =========================
class BTVisualizer:
    def __init__(self, tree: BTNode, transparent=False):
        self.tree = tree
        self.transparent = transparent
        self.lines = []
    def to_dot(self):
        self.lines.append("digraph BehaviorTree {")
        self.lines.append("  rankdir=TB;")
        self.lines.append("  node [fontname=Helvetica];")
        if self.transparent:
            self.lines.append("  graph [bgcolor=transparent];")
        else:
            self.lines.append("  graph [bgcolor=white];")
        self._visit(self.tree)
        self.lines.append("}")
        return "\n".join(self.lines)
    def _visit(self, node: BTNode):
        style = self._node_style(node)
        self.lines.append(f'  "{node.id}" [label="{node.label()}" {style}];')
        for child in node.children:
            self.lines.append(f'  "{node.id}" -> "{child.id}";')
            self._visit(child)
    def _node_style(self, node: BTNode):
        if isinstance(node, Sequence): return 'shape=box style="filled,rounded" fillcolor=lightblue'
        if isinstance(node, Selector): return 'shape=box style="filled,rounded" fillcolor=khaki'
        if isinstance(node, Condition): return 'shape=ellipse style="filled" fillcolor=palegreen'
        if isinstance(node, Action): return 'shape=ellipse style="filled" fillcolor=lightcoral'
        return 'shape=box'

def save_dot(tree, filename, transparent=False):
    dot = BTVisualizer(tree, transparent=transparent).to_dot()
    with open(filename,"w") as f: f.write(dot)
    print(f"[OK] Behavior tree saved to {filename}")

# =========================
# Batch Generation
# =========================
OUTPUT_FOLDER = "output"
DOT_FOLDER = os.path.join(OUTPUT_FOLDER, "dot")
PNG_FOLDER = os.path.join(OUTPUT_FOLDER, "png")
os.makedirs(DOT_FOLDER, exist_ok=True)
os.makedirs(PNG_FOLDER, exist_ok=True)

for i in range(args.num):
    blackboard = dict(BLACKBOARD)
    tree = generate_random_tree(args.max_depth)
    result = tree.tick(blackboard)
    print(f"Tree {i+1} tick result:", result)

    dot_path = os.path.join(DOT_FOLDER, f"bt_{i+1}.dot")
    save_dot(tree, filename=dot_path, transparent=args.transparent)

    png_path = os.path.join(PNG_FOLDER, f"bt_{i+1}.png")
    cmd = [args.layout, "-Tpng", dot_path, "-o", png_path]
    if args.transparent:
        cmd.append("-Gbgcolor=transparent")
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Saved image {png_path}")
    except Exception as e:
        print(f"[ERROR] Could not generate image: {e}")
