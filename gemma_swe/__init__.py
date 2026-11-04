"""gemma_swe: post-training + submission tooling for the Gemma 4 Developer Agent competition.

Pure-Python modules (no GPU / no transformers needed):
    constants, tool_schema, calibration, harness_prompt, observations, patch_utils,
    graph_utils, records, remap_openhands, traj_filters, adapter_utils, proxy, harvest,
    submission_builder

GPU / transformers modules (imported lazily by the training scripts):
    chat_render, losses, dequant
"""

__version__ = "0.1.0"
