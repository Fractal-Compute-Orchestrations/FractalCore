import torch
from slicer.contracts import IngestInput, IngestOutput

try:
    import transformers
    from transformers import AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class LlamaDecoderLayerMock(torch.nn.Module):
    """
    Mock representation of a single Llama 3 (8B) decoder layer
    to run slicing pipelines on platforms without HF models cached
    or without transformers package.
    """
    def __init__(self, dim: int = 4096, hidden_dim: int = 14336):
        super().__init__()
        # Attention Projections (GQA config: 32 query heads, 8 key/value heads)
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim // 4, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim // 4, bias=False)
        self.o_proj = torch.nn.Linear(dim, dim, bias=False)
        
        # SwiGLU MLP Block
        self.gate_proj = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = torch.nn.Linear(hidden_dim, dim, bias=False)
        
        # Layer Norms
        self.input_layernorm = torch.nn.LayerNorm(dim)
        self.post_attention_layernorm = torch.nn.LayerNorm(dim)

    def forward(self, x, kv_cache=None):
        h = self.input_layernorm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        attn = self.o_proj(q)
        x = x + attn
        
        h2 = self.post_attention_layernorm(x)
        ffn = self.down_proj(torch.nn.functional.silu(self.gate_proj(h2)) * self.up_proj(h2))
        return x + ffn

def load_model(model_id: str):
    """Loads model configurations and references (runs once)."""
    print(f"[Workstation 1] Loading raw model model_id: '{model_id}'...")
    return {"model_id": model_id}

def isolate_layer(monolithic_model, layer_idx: int) -> IngestOutput:
    """Isolates a target transformer layer block module."""
    model_id = monolithic_model["model_id"]
    print(f"[Workstation 1] Isolating Layer {layer_idx} from model '{model_id}'...")
    
    if HAS_TRANSFORMERS:
        try:
            print(f"[Workstation 1] Using transformers to load layer {layer_idx} from '{model_id}'...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="cpu", 
                torch_dtype=torch.float16
            )
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                isolated_layer = model.model.layers[layer_idx]
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                isolated_layer = model.transformer.h[layer_idx]
            else:
                raise AttributeError("Could not identify decoder block list in this model structure.")
        except Exception as e:
            print(f"[Workstation 1] Real load failed ({e}). Falling back to simulation model.")
            isolated_layer = LlamaDecoderLayerMock()
    else:
        print("[Workstation 1] Transformers not found. Using Llama 3 (8B) simulation layer module.")
        isolated_layer = LlamaDecoderLayerMock()
        
    param_count = sum(p.numel() for p in isolated_layer.parameters())
    
    return IngestOutput(
        layer_index=layer_idx,
        module=isolated_layer,
        original_params=param_count
    )
