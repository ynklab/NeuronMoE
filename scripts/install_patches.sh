#!/bin/bash
# Install patches for LLaMA-Factory, transformers, and lm-evaluation-harness
# Run from the NeuronMoE root directory

set -e

NEURONMOE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

apply_patch_file() {
    local target_dir="$1"
    local patch_file="$2"
    local label="$3"

    if [ ! -d "$target_dir" ]; then
        echo "Missing target directory for $label: $target_dir"
        exit 1
    fi

    if [ ! -f "$patch_file" ]; then
        echo "Missing patch file for $label: $patch_file"
        exit 1
    fi

    if patch --dry-run -N -d "$target_dir" -p1 < "$patch_file" >/dev/null; then
        patch -N -d "$target_dir" -p1 < "$patch_file"
    elif patch --dry-run -R -d "$target_dir" -p1 < "$patch_file" >/dev/null; then
        echo "  $label already applied; skipping."
    else
        echo "Failed to apply $label. Check package versions and patch compatibility."
        exit 1
    fi
}

# Resolve site-packages directory cleanly
SITE_PKG=$(uv run python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null | tail -n 1)

if [ -z "$SITE_PKG" ] || [ ! -d "$SITE_PKG" ]; then
    echo "Could not resolve site-packages directory."
    exit 1
fi

# ===== LLaMA-Factory patches =====
echo "Applying LLaMA-Factory patches..."
apply_patch_file "$SITE_PKG/llmtuner" "$NEURONMOE_DIR/patches/llama_factory.patch" "LLaMA-Factory patch"
# Copy new files added by the MoE patch (the .patch above only modifies existing files).
cp -r "$NEURONMOE_DIR/patches/llama_factory_files/." "$SITE_PKG/llmtuner/"
echo "  Done."

# ===== Transformers patch =====
echo "Applying transformers patches..."
apply_patch_file "$SITE_PKG/transformers" "$NEURONMOE_DIR/patches/transformers/modeling_llama.patch" "transformers LLaMA patch"
apply_patch_file "$SITE_PKG/transformers" "$NEURONMOE_DIR/patches/transformers/trainer.patch" "transformers Trainer patch"
echo "  Done."

# ===== lm-evaluation-harness patches =====
echo "Applying lm-evaluation-harness patches..."
if [ ! -d "$SITE_PKG/lm_eval/tasks" ]; then
    echo "Missing lm-evaluation-harness tasks directory: $SITE_PKG/lm_eval/tasks"
    exit 1
fi
cp -r "$NEURONMOE_DIR/patches/lm_eval_tasks/mmlu_el" "$SITE_PKG/lm_eval/tasks/"
cp -r "$NEURONMOE_DIR/patches/lm_eval_tasks/mmlu_tr" "$SITE_PKG/lm_eval/tasks/"
echo "  Done."

echo "All patches applied successfully."
