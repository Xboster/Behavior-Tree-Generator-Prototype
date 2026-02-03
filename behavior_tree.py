"""
Behavior Tree Prototype (Single File)

Requirements:
- Python 3.9+
- Graphviz installed (for visualization)

Usage:
    python behavior_tree.py
    dot -Tpng bt.dot -o bt.png
"""


from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import random
import uuid


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
    def __init__(self, name, condition_fn):
        super().__init__(name)
        self.condition_fn = condition_fn

    def tick(self, blackboard):
        self.last_status = (
            Status.SUCCESS if self.condition_fn(blackboard) else Status.FAILURE
        )
        return self.last_status


class Action(BTNode):
    def __init__(self, name, action_fn):
        super().__init__(name)
        self.action_fn = action_fn

    def tick(self, blackboard):
        self.last_status = self.action_fn(blackboard)
        return self.last_status

# =========================
# Grammar-Based Generation
# =========================

GRAMMAR = {
    "NODE": [("CONTROL", 0.8), ("LEAF", 0.2)],
    "CONTROL": [("SEQUENCE", 0.5), ("SELECTOR", 0.5)],
    "SEQUENCE": [("Sequence", ["NODE"])],  # children added randomly
    "SELECTOR": [("Selector", ["NODE"])],
    "LEAF": [("CONDITION", 0.5), ("ACTION", 0.5)],
}


def weighted_choice(options):
    symbols, weights = zip(*options)
    return random.choices(symbols, weights=weights)[0]

def expand(symbol, depth, max_depth):
    if depth >= max_depth:
        # Force a leaf if max depth reached
        return (weighted_choice([("CONDITION", 0.5), ("ACTION", 0.5)]),)

    if symbol == "NODE":
        # Chance to branch as control or terminate as leaf
        leaf_prob = min(0.1 + depth * 0.2, 0.9)
        control_prob = 1 - leaf_prob
        choice = weighted_choice([("CONTROL", control_prob), ("LEAF", leaf_prob)])
        return expand(choice, depth + 1, max_depth)

    if symbol == "CONTROL":
        # Pick SEQUENCE or SELECTOR
        choice = weighted_choice([("SEQUENCE", 0.5), ("SELECTOR", 0.5)])
        return expand(choice, depth + 1, max_depth)

    if symbol in ("SEQUENCE", "SELECTOR"):
        node_type, _ = GRAMMAR[symbol][0]
        num_children = random.randint(1, 3)
        return (
            node_type,
            [expand("NODE", depth + 1, max_depth) for _ in range(num_children)]
        )

    if symbol == "LEAF":
        return expand(weighted_choice([("CONDITION", 0.5), ("ACTION", 0.5)]), depth + 1, max_depth)

    # Terminal nodes (CONDITION/ACTION)
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

    if node_type == "Sequence":
        node = Sequence("Sequence")
        for child in ast[1]:
            node.add_child(build_bt(child))
        return node

    if node_type == "Selector":
        node = Selector("Selector")
        for child in ast[1]:
            node.add_child(build_bt(child))
        return node

    if node_type == "CONDITION":
        name, fn = random.choice(CONDITIONS)
        return Condition(name, fn)

    if node_type == "ACTION":
        name, fn = random.choice(ACTIONS)
        return Action(name, fn)
    
    if node_type == "LEAF":
        return build_bt((weighted_choice(GRAMMAR["LEAF"]),))

    raise ValueError(f"Unknown AST node: {ast}")

def generate_random_tree(max_depth=5):
    # Force the root to be a CONTROL node (Sequence or Selector)
    root_symbol = weighted_choice([("SEQUENCE", 0.5), ("SELECTOR", 0.5)])
    ast = expand(root_symbol, depth=0, max_depth=max_depth)
    root = build_bt(ast)
    return BehaviorTree(root)


# =========================
# Example / Test Harness
# =========================

if __name__ == "__main__":
    # Blackboard (game state)
    blackboard = {
        "enemy_visible": True,
        "ammo": 10
    }

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
    tree = generate_random_tree(max_depth=7)

    # Tick tree
    print("Tick result:", tree.tick(blackboard))

    # Visualize
    save_dot(tree)
