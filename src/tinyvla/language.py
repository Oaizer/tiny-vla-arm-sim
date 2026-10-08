"""Check commands and list the words and objects the model supports."""
from dataclasses import dataclass
import re
import numpy as np

OBJECTS = ('red cube', 'red cylinder', 'blue cube', 'green cylinder')
PADS = ('yellow', 'purple', 'cyan')
TASKS = ('place', 'lift')
VOCAB = ('<pad>', 'put', 'place', 'move', 'the', 'red', 'blue', 'green', 'cube',
         'cylinder', 'on', 'onto', 'to', 'yellow', 'purple', 'cyan', 'pad', 'target',
         'lift', 'raise', 'pick', 'up')

@dataclass(frozen=True)
class Task:
    kind: str
    object: str
    pad: str | None
    instruction: str

def parse(instruction):
    text=' '.join(instruction.lower().strip().rstrip('.').split())
    m=re.fullmatch(r'(?:put|place|move) (?:the )?((?:red|blue|green) (?:cube|cylinder)) (?:on|onto|to) (?:the )?(yellow|purple|cyan) (?:pad|target)',text)
    if m:kind,obj,pad='place',m[1],m[2]
    else:
        m=re.fullmatch(r'(?:lift|raise|pick up) (?:the )?((?:red|blue|green) (?:cube|cylinder))',text)
        if not m:raise ValueError('I cannot follow that command. Try "put the red cube on the cyan pad" or "lift the red cylinder". See docs/MODEL_GOAL.md for the full list.')
        kind,obj,pad='lift',m[1],None
    if obj not in OBJECTS:raise ValueError(f'The model does not support {obj!r}. Available objects: {OBJECTS}')
    return Task(kind,obj,pad,text)

def tokenize(instruction):
    text=parse(instruction).instruction
    words=text.split()
    if len(words)>16:raise ValueError('The command is too long. Use 16 words or fewer.')
    return np.array([VOCAB.index(w) for w in words]+[0]*(16-len(words)),np.int64)

def commands(heldout=False):
    result=[]
    for obj in OBJECTS:
        for pad in PADS:
            templates=([f'move {obj} onto {pad} target',f'place {obj} on {pad} pad'] if heldout else
                       [f'put the {obj} on the {pad} pad',f'place the {obj} onto the {pad} target',f'move the {obj} to the {pad} pad'])
            result.extend(templates)
        result.extend([f'pick up {obj}',f'raise {obj}'] if heldout else [f'lift the {obj}',f'raise the {obj}',f'pick up the {obj}'])
    return result
