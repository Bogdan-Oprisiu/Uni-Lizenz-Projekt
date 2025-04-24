# ----------------------------- teacher_v3/utils_v3.py -----------------------------
"""Utility functions for the LoRA-based teacher pipeline (v3)."""
from __future__ import annotations

import gzip
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Optional, Any

import torch

# --- Make sure local modules are importable ---
# Assuming this file is in teacher_v3/
MODULE_DIR = Path(__file__).resolve().parent  # .../teacher_v3
ROOT_DIR = MODULE_DIR.parent  # .../LLM (project root)
for p in (MODULE_DIR, ROOT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Now import config_v3 after adjusting sys.path
try:
    import config_v3
except ImportError:
    print("ERROR: Could not import config_v3.py. Ensure it's in the same directory as utils_v3.py.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Tokenizer Loader
# ---------------------------------------------------------------------------

def load_tokenizer(use_fast: bool = True):
    """
    Loads the tokenizer, prioritizing the one saved with the LoRA adapter.

    Loading Order:
    1. Try loading from the LORA_ADAPTER_DIR.
    2. Fallback to loading from the BASE_MODEL_ID.
    3. Final fallback to the explicitly defined TOKENIZER_PATH (wrapping if necessary).

    Args:
        use_fast (bool): Whether to prefer fast tokenizers.

    Returns:
        transformers.PreTrainedTokenizer: The loaded tokenizer instance.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from tokenizers import Tokenizer as TokenizersTokenizer  # Avoid name clash

    adapter_path = config_v3.LORA_ADAPTER_DIR.resolve()  # Use resolved path
    base_model_id = config_v3.BASE_MODEL_ID
    explicit_tokenizer_path = config_v3.TOKENIZER_PATH.resolve()  # Use resolved path

    # --- Attempt 1: Load from Adapter Directory ---
    try:
        print(f"Attempting to load tokenizer from adapter directory: {adapter_path}")
        # PEFT usually saves the tokenizer alongside the adapter weights
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=use_fast)
        print("✅ Tokenizer loaded from adapter directory.")
        return tokenizer
    except OSError:
        print(f"INFO: Tokenizer not found in adapter directory '{adapter_path}'.")
    except Exception as e:
        print(f"WARNING: Unexpected error loading tokenizer from adapter dir: {e}")

    # --- Attempt 2: Load from Base Model ID ---
    try:
        print(f"Attempting to load tokenizer from base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=use_fast)
        print(f"✅ Tokenizer loaded from base model '{base_model_id}'.")
        return tokenizer
    except OSError:
        print(f"INFO: Tokenizer not found for base model ID '{base_model_id}'.")
    except Exception as e:
        print(f"WARNING: Unexpected error loading tokenizer from base model ID: {e}")

    # --- Attempt 3: Load from Explicit Path (and wrap if needed) ---
    if explicit_tokenizer_path.exists():
        print(f"Attempting fallback to explicit path: {explicit_tokenizer_path}")
        try:
            # Check if it's a directory (HuggingFace format) or a file (tokenizers.json format)
            if explicit_tokenizer_path.is_dir():
                tokenizer = AutoTokenizer.from_pretrained(str(explicit_tokenizer_path), use_fast=use_fast)
                print(f"✅ Tokenizer loaded from explicit directory path: {explicit_tokenizer_path}")
                return tokenizer
            elif explicit_tokenizer_path.is_file() and explicit_tokenizer_path.suffix == ".json":
                # Load raw tokenizers.Tokenizer and wrap it
                print("   (Detected .json file, attempting to wrap with PreTrainedTokenizerFast)")
                raw_tok = TokenizersTokenizer.from_file(str(explicit_tokenizer_path))
                # Define special tokens - adjust these based on your actual tokenizer/model needs
                # Common T5 tokens: </s>, <pad>, <unk>
                # Ensure these match what the model expects/was trained with
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=raw_tok,
                    bos_token="<s>",  # Example BOS token, T5 might not use one explicitly
                    eos_token="</s>",
                    unk_token="<unk>",
                    pad_token="<pad>",  # T5 often uses <pad>
                    # Add other special tokens if needed (e.g., cls_token, sep_token, mask_token)
                )
                print(f"✅ Tokenizer loaded and wrapped from explicit file path: {explicit_tokenizer_path}")
                return tokenizer
            else:
                print(
                    f"WARNING: Explicit tokenizer path '{explicit_tokenizer_path}' is neither a valid directory nor a .json file.")

        except Exception as wrap_e:
            print(f"ERROR: Failed to load/wrap tokenizer from explicit path '{explicit_tokenizer_path}': {wrap_e}")
            # Continue to final error if this fails
    else:
        print(f"INFO: Explicit tokenizer path not found: {explicit_tokenizer_path}")

    # --- Final Error ---
    print("\n❌ ERROR: Failed to load tokenizer using all available methods:")
    print(f"   - Adapter Path: {adapter_path} (Not found or error)")
    print(f"   - Base Model ID: {base_model_id} (Not found or error)")
    print(f"   - Explicit Path: {explicit_tokenizer_path} (Not found or error)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Teacher Model Loader (Base + LoRA Adapter)
# ---------------------------------------------------------------------------

def load_teacher_model(
        device: Optional[str] = None,
        for_training: bool = False  # Set to True during fine-tuning to load base model only initially
) -> torch.nn.Module:
    """
    Loads the base model and applies the LoRA adapter if available and not loading for training start.
    Handles device placement and optional quantization/dtype for the base model.

    Args:
        device (Optional[str]): Target device (e.g., "cuda", "cpu"). Defaults to config.
        for_training (bool): If True, loads only the base model initially, assuming
                             PEFT `get_peft_model` will be called later. If False (inference),
                             attempts to load the base model and apply the adapter.

    Returns:
        torch.nn.Module: The loaded model (either base or base + adapter).
    """
    from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig  # Import PeftModel

    target_device = device or config_v3.DEVICE
    # Resolve paths to be absolute
    adapter_path = config_v3.LORA_ADAPTER_DIR.resolve()
    base_model_id = config_v3.BASE_MODEL_ID
    use_bnb = config_v3.USE_BNB_INT8_BASE
    model_dtype_str = config_v3.MODEL_DTYPE

    print(f"\n--- Loading Teacher Model ---")
    print(f"Base Model ID: {base_model_id}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Target Device: {target_device}")
    print(f"Use 8-bit (Base): {use_bnb}")
    print(f"Model Dtype: {model_dtype_str}")
    print(f"Loading for training init: {for_training}")
    print("-" * 29)

    # Determine torch dtype
    torch_dtype = None
    if model_dtype_str == "float16":
        torch_dtype = torch.float16
    elif model_dtype_str == "bfloat16":
        # Check if bf16 is supported
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("INFO: bfloat16 is supported and selected.")
        else:
            print("WARNING: bfloat16 not supported on this hardware, falling back to float32.")
            torch_dtype = torch.float32  # Fallback dtype
    else:
        torch_dtype = torch.float32  # Default if None or other string
        print("INFO: Using float32 dtype.")

    # --- Quantization Config (for base model if USE_BNB_INT8_BASE is True) ---
    quantization_config = None
    if use_bnb:
        try:
            # Ensure bitsandbytes is installed
            import bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # Optional: Add other bnb config args if needed
                # bnb_8bit_compute_dtype=torch.float16, # Example
            )
            print("INFO: BitsandBytes 8-bit quantization enabled for base model loading.")
            # Note: 8-bit loading often requires device_map="auto" or specific device placement
            if target_device == "cpu":
                print("WARNING: BitsAndBytes quantization is typically used with CUDA.")

        except ImportError:
            print("⚠️ WARNING: bitsandbytes not installed, but USE_BNB_INT8_BASE=True. Ignoring quantization.")
            use_bnb = False  # Disable if import fails
        except Exception as e:
            print(f"⚠️ WARNING: Error setting up BitsAndBytesConfig: {e}. Ignoring quantization.")
            use_bnb = False

    # --- Device Mapping ---
    # Use "auto" for multi-GPU or when using quantization effectively
    device_map = "auto" if (target_device.startswith("cuda") or target_device == "auto") else None
    if target_device == "cpu":
        device_map = "cpu"
        # Quantization often doesn't work well or at all on CPU
        if use_bnb:
            print("WARNING: Disabling device_map='auto' for CPU. Quantization might not be effective.")
            # device_map = None # Or keep as 'cpu' if AutoModel handles it

    print(f"Calculated device_map: {device_map}")

    # --- Load Base Model ---
    print(f"🔄 Loading base model: {base_model_id}")
    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,  # Pass None if not use_bnb
            device_map=device_map,
            torch_dtype=torch_dtype if not use_bnb else None,  # BNB handles dtype for quantized layers
            low_cpu_mem_usage=True if device_map == "auto" else False,  # Optimization for large models
            # Add other necessary kwargs like trust_remote_code=True if needed
        )
        print("✅ Base model loaded.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load base model '{base_model_id}': {e}")
        # Consider printing traceback for more details
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # If loading specifically for starting the training process, return base model now
    if for_training:
        print("INFO: Returning base model for training initialization (PEFT adapter will be applied later).")
        # No need to set eval() mode here
        return base_model

    # --- Load and Apply LoRA Adapter (for inference or continuing training) ---
    adapter_config_file = adapter_path / "adapter_config.json"
    if adapter_config_file.exists():
        print(f"🔄 Found LoRA adapter config at: {adapter_path}")
        try:
            # Load the PEFT model by applying the adapter to the base model
            print("   Applying LoRA adapter...")
            # Ensure the base model is on the target device *before* loading adapter if not using device_map='auto'
            # if device_map is None and target_device != 'cpu':
            #    base_model.to(target_device)

            # is_trainable=False is generally recommended for inference
            model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
            print("✅ LoRA adapter loaded and applied successfully.")
            # Merge adapter for potentially faster inference (optional)
            # print("   Merging LoRA adapter...")
            # model = model.merge_and_unload()
            # print("   Adapter merged.")

            model.eval()  # Set to evaluation mode for inference
            return model
        except Exception as e:
            print(f"⚠️ WARNING: Failed to load or apply LoRA adapter from '{adapter_path}': {e}")
            print("         Proceeding with the base model only.")
            base_model.eval()  # Set base model to eval mode
            return base_model  # Return base model as fallback
    else:
        print(f"INFO: No LoRA adapter found at '{adapter_path}'. Using base model for inference.")
        base_model.eval()  # Set base model to eval mode
        return base_model  # Return base model if no adapter exists


# ---------------------------------------------------------------------------
# 3. Batching Helper
# ---------------------------------------------------------------------------

def batched(iterable: Iterable[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from iterable."""
    if n < 1:
        raise ValueError("Batch size n must be at least 1")
    batch: List[Any] = []
    iterator = iter(iterable)
    while True:
        try:
            item = next(iterator)
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        except StopIteration:
            if batch:
                yield batch
            break


# ---------------------------------------------------------------------------
# 4. Soft Probabilities Helper
# ---------------------------------------------------------------------------

def extract_topk(
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0
) -> List[Tuple[int, float]]:
    """
    Extracts top-k token IDs and their probabilities from logits,
    applying temperature scaling. Returns probabilities in float16 for space.

    Args:
        logits (torch.Tensor): Raw logits output from the model for a single token position.
                               Shape should be (vocab_size,).
        k (int): The number of top probabilities to return.
        temperature (float): Temperature for scaling logits. Must be positive.

    Returns:
        List[Tuple[int, float]]: A list of (token_id, probability) tuples.
    """
    if temperature <= 0:
        warnings.warn(f"Temperature must be positive, but got {temperature}. Setting to 1.0.")
        temperature = 1.0

    # Ensure logits are on CPU and float32 for stable softmax
    logits_cpu = logits.detach().float().cpu()

    # Apply temperature scaling
    scaled_logits = logits_cpu / temperature
    # Calculate probabilities using softmax
    probs = torch.softmax(scaled_logits, dim=-1)
    # Get top k probabilities and their indices
    top_k_probs, top_k_indices = torch.topk(probs, k)

    # Convert probabilities to float16 and create list of tuples
    return [
        (int(idx_item.item()), float(prob_item.half().item()))  # Store as float16
        for idx_item, prob_item in zip(top_k_indices, top_k_probs)
    ]


# ---------------------------------------------------------------------------
# 5. Lightweight I/O Helpers
# ---------------------------------------------------------------------------

def open_writable(path: Path, binary: bool = False, compress: bool = False) -> Any:
    """Opens a file for writing, creating parent dirs, handling gzip."""
    resolved_path = path.resolve()  # Resolve path for accurate parent creation
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "wt"
    encoding = None if binary else "utf-8"

    # Adjust path suffix if compression is requested but not in name
    if compress and not resolved_path.name.endswith(".gz"):
        resolved_path = resolved_path.with_name(resolved_path.name + ".gz")

    if resolved_path.name.endswith(".gz"):
        return gzip.open(resolved_path, mode=mode, encoding=encoding)
    else:
        # Ensure compression isn't True if extension isn't .gz
        if compress:
            warnings.warn(
                f"Compression requested but file path '{resolved_path}' does not end with .gz. Saving uncompressed.")
        return open(resolved_path, mode=mode, encoding=encoding)


def write_jsonl(fh: Any, data: Any) -> None:
    """Writes a single Python object as a JSON line to the file handle."""
    try:
        # Ensure file handle is open in text mode
        if hasattr(fh, 'mode') and 'b' in fh.mode:
            raise TypeError("File handle must be opened in text mode ('wt') for write_jsonl")
        fh.write(json.dumps(data, ensure_ascii=False) + "\n")
    except TypeError as e:
        print(f"ERROR: Failed to serialize data to JSON: {data}")
        print(f"       Error: {e}")
        # Decide if you want to raise the error or just log it
        # raise
    except Exception as e:
        print(f"ERROR: Unexpected error writing JSONL line: {e}")
        # raise


def save_json(data: Any, path: Path, compress: bool = False, indent: Optional[int] = None) -> None:
    """Saves data to a JSON file (potentially gzipped)."""
    resolved_path = path.resolve()
    target_path_str = str(resolved_path) + (".gz" if compress and not resolved_path.name.endswith(".gz") else "")
    print(f"💾 Saving JSON data to: {target_path_str}")
    try:
        with open_writable(resolved_path, compress=compress) as fh:
            json.dump(data, fh, ensure_ascii=False, indent=indent)
        print("✅ Save complete.")
    except Exception as e:
        print(f"❌ ERROR: Failed to save JSON to {target_path_str}: {e}")
        # raise


def load_json(path: Path) -> Any:
    """Loads data from a JSON file (handles gzip)."""
    resolved_path = path.resolve()
    print(f"💾 Loading JSON data from: {resolved_path}")
    is_gzipped = resolved_path.name.endswith(".gz")
    mode = "rb" if is_gzipped else "rt"
    encoding = None if is_gzipped else "utf-8"
    opener = gzip.open if is_gzipped else open

    if not resolved_path.exists():
        print(f"❌ ERROR: File not found: {resolved_path}")
        raise FileNotFoundError(f"File not found: {resolved_path}")

    try:
        with opener(resolved_path, mode=mode, encoding=encoding) as fh:
            data = json.load(fh)
        print("✅ Load complete.")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in file {resolved_path}: {e}")
        raise
    except Exception as e:
        print(f"❌ ERROR: Failed to load JSON from {resolved_path}: {e}")
        raise


def load_text_file(path: Path) -> List[str]:
    """Loads lines from a text file, stripping whitespace and skipping empty lines."""
    resolved_path = path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Text file not found: {resolved_path}")
    print(f"💾 Loading text lines from: {resolved_path}")
    lines = []
    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(lines)} non-empty lines.")
        return lines
    except Exception as e:
        print(f"❌ ERROR: Failed to read text file {resolved_path}: {e}")
        raise


def load_jsonl_file(path: Path) -> List[Any]:
    """Loads data from a JSONL file (one JSON object per line), handling gzip."""
    resolved_path = path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {resolved_path}")

    print(f"💾 Loading JSONL data from: {resolved_path}")
    data = []
    is_gzipped = resolved_path.name.endswith(".gz")
    opener = gzip.open if is_gzipped else open
    mode = "rt"  # Always text mode for jsonl
    encoding = "utf-8"
    line_count = 0
    error_count = 0

    try:
        with opener(resolved_path, mode=mode, encoding=encoding) as f:
            for i, line in enumerate(f):
                line_count = i + 1
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        error_count += 1
                        print(f"ERROR: Invalid JSON on line {line_count} in {resolved_path}: {e}")
                        print(f"       Line content: {line[:200]}{'...' if len(line) > 200 else ''}")
                        # Decide how to handle errors: skip, raise, etc.
                        # raise # Uncomment to make errors fatal
    except Exception as e:
        print(f"❌ ERROR: Failed to read JSONL file {resolved_path}: {e}")
        raise

    print(f"✅ Loaded {len(data)} objects from {line_count} lines.")
    if error_count > 0:
        print(f"⚠️ Found {error_count} lines with JSON parsing errors.")
    return data
