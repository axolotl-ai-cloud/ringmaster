from ringmaster.strategies.state_passing import is_recurrent_mixer, wire_recurrent_layers
from ringmaster.strategies.ulysses import register_ulysses, REGISTERED_NAME
from ringmaster.strategies.usp import auto_select

__all__ = [
    "register_ulysses",
    "REGISTERED_NAME",
    "auto_select",
    "is_recurrent_mixer",
    "wire_recurrent_layers",
]
