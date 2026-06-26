from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class IngestInput:
    model_id_or_path: str
    layer_index: int

@dataclass(frozen=True)
class IngestOutput:
    layer_index: int
    module: torch.nn.Module
    original_params: int

@dataclass(frozen=True)
class QuantizeInput:
    layer_index: int
    module: torch.nn.Module

@dataclass(frozen=True)
class QuantizeOutput:
    layer_index: int
    module: torch.nn.Module
    in_memory_size_mb: float

@dataclass(frozen=True)
class TraceInput:
    layer_index: int
    module: torch.nn.Module

@dataclass(frozen=True)
class TraceOutput:
    layer_index: int
    exported_program: object  # Can be ExportedProgram or mock object

@dataclass(frozen=True)
class DelegateInput:
    layer_index: int
    exported_program: object

@dataclass(frozen=True)
class DelegateOutput:
    layer_index: int
    delegated_program: object  # Can be EdgeProgramManager or mock object

@dataclass(frozen=True)
class ExportInput:
    layer_index: int
    delegated_program: object
    output_path: str

@dataclass(frozen=True)
class ExportOutput:
    layer_index: int
    file_path: str
    file_size_bytes: int
