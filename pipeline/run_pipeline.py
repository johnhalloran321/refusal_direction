import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)

        # Test the new refusal_toks
        print(f"\nTesting refusal detection:")
        print(f"Refusal tokens: {model_base.refusal_toks}")
        print(f"Decoded: {[model_base.tokenizer.decode([tok]) for tok in model_base.refusal_toks]}")

        # Manually check: does "I'm sorry" appear in the vocab at position we're checking?
        test_response = "I'm sorry, but I can't assist with that."
        test_tokens = model_base.tokenizer.encode(test_response, add_special_tokens=False)
        print(f"\nTest response tokens: {test_tokens[:5]}")
        print(f"Decoded: {[model_base.tokenizer.decode([tok]) for tok in test_tokens[:5]]}")
        print(f"Any overlap with refusal_toks? {set(test_tokens) & set(model_base.refusal_toks)}")
        # exit(-1)
        # ADD THIS DEBUG SECTION
        print("\n" + "="*80)
        print("BASELINE REFUSAL SCORES")
        print("="*80)
        print(f"Harmful instructions - Mean: {harmful_train_scores.mean():.4f}, Min: {harmful_train_scores.min():.4f}, Max: {harmful_train_scores.max():.4f}")
        print(f"NaN in harmful: {torch.isnan(harmful_train_scores).sum()}/{len(harmful_train_scores)}")
        print("="*80 + "\n")

        # Test what model actually generates
        print("\n" + "="*80)
        print("SAMPLE RESPONSES TO HARMFUL INSTRUCTIONS")
        print("="*80)
        test_samples = model_base.generate_completions(
            [{'instruction': inst, 'category' : 'harmful'} for inst in harmful_train[:5]],
            fwd_pre_hooks=[],
            fwd_hooks=[],
            max_new_tokens=50
        )
        for i, sample in enumerate(test_samples):
            print(f"\nSample {i+1}:")
            print(f"  Instruction: {sample['prompt'][:80]}...")
            print(f"  Response: {sample['response'][:150]}...")
            print(f"  Refusal score: {harmful_train_scores[i]:.4f}")
        print("="*80 + "\n")

        # exit(-1)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val
def get_orthogonalized_matrix(matrix, direction):
    direction = direction.to(dtype=matrix.dtype, device=matrix.device)
    proj = matrix.T @ direction
    return matrix - torch.outer(direction, proj)

def orthogonalize_and_save_model(cfg, model_base, direction, layers_to_modify=None):
    from tqdm import tqdm
    
    # Access actual layer objects (not module references)
    all_layers = model_base.model.model.layers  # Adjust if different
    
    if layers_to_modify is None:
        layers_to_modify = range(len(all_layers))
    
    direction = direction.to(model_base.model.device).squeeze()
    direction = direction / direction.norm()
    
    modified = 0
    for idx in tqdm(list(layers_to_modify)):
        layer = all_layers[idx]
        
        # Orthogonalize attention output (adjust attribute names if needed)
        if hasattr(layer.self_attn, 'o_proj'):
            layer.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
                layer.self_attn.o_proj.weight.data, direction
            )
            modified += 1
        
        # Orthogonalize MLP output
        if hasattr(layer.mlp, 'down_proj'):
            layer.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                layer.mlp.down_proj.weight.data, direction
            )
            modified += 1
    
    print(f"\nModified {modified} weights")
    
    # Verify
    W = all_layers[len(all_layers)//2].mlp.down_proj.weight.data
    direction_verify = direction.to(dtype=W.dtype)
    print(f"Verification: {torch.abs(W.T @ direction_verify).max():.6f}")    
    # print(f"Verification: {torch.abs(W.T @ direction).max():.6f}  ")
    
    # Save
    output_dir = os.path.join(cfg.artifact_path(), 'abliterated_model')
    os.makedirs(output_dir, exist_ok=True)
    model_base.model.save_pretrained(output_dir, safe_serialization=True)
    model_base.tokenizer.save_pretrained(output_dir)
    return output_dir
# def get_orthogonalized_matrix(matrix, direction):
#     """Orthogonalize matrix columns with respect to direction."""
#     direction = direction / direction.norm()
#     proj = matrix.T @ direction
#     return matrix - torch.outer(direction, proj)

# def orthogonalize_and_save_model(cfg, model_base, direction, layers_to_modify=None):
#     """Permanently orthogonalize model weights and save."""
#     from tqdm import tqdm
    
#     if layers_to_modify is None:
#         layers_to_modify = range(len(model_base.model_block_modules))
    
#     device = model_base.model.device
#     direction = direction.to(device).to(model_base.model.dtype).squeeze()
#     direction = direction / direction.norm()
    
#     for layer_idx in tqdm(list(layers_to_modify), desc="Orthogonalizing"):
#         attn_module = model_base.model_attn_modules[layer_idx]
#         mlp_module = model_base.model_mlp_modules[layer_idx]
        
#         if hasattr(attn_module, 'weight'):
#             attn_module.weight.data = get_orthogonalized_matrix(
#                 attn_module.weight.data, direction
#             )
        
#         if hasattr(mlp_module, 'weight'):
#             mlp_module.weight.data = get_orthogonalized_matrix(
#                 mlp_module.weight.data, direction
#             )
    
#     # Verify
#     test_module = model_base.model_mlp_modules[len(list(layers_to_modify))//2]
#     if hasattr(test_module, 'weight'):
#         dots = torch.abs(test_module.weight.data.T @ direction).max().item()
#         print(f"Verification: max |col·direction| = {dots:.6f}")
    
#     # Save
#     output_dir = os.path.join(cfg.artifact_path(), 'abliterated_model')
#     os.makedirs(output_dir, exist_ok=True)
#     model_base.model.save_pretrained(output_dir, safe_serialization=True)
#     model_base.tokenizer.save_pretrained(output_dir)
    
#     return output_dir
# def orthogonalize_and_save_model(cfg, model_base, direction, layers_to_modify=None):
#     """
#     Permanently orthogonalize model weights and save the modified model.
    
#     This applies the same orthogonalization that hooks would apply temporarily,
#     but directly modifies the weight matrices permanently.
#     """
#     # Get the orthogonalization modification function from ModelBase
#     ortho_mod_fn = model_base._get_orthogonalization_mod_fn(direction)
    
#     # Determine which layers to modify
#     if layers_to_modify is None:
#         # Apply to all layers (same as get_all_direction_ablation_hooks does)
#         layers_to_modify = range(len(model_base.model_block_modules))
    
#     print(f"Orthogonalizing layers: {list(layers_to_modify)}")
    
#     # Move direction to model device
#     device = model_base.model.device
#     direction = direction.to(device)
    
#     # Apply orthogonalization to each layer's weights
#     for layer_idx in layers_to_modify:
#         # Get attention and MLP modules for this layer
#         attn_module = model_base.model_attn_modules[layer_idx]
#         mlp_module = model_base.model_mlp_modules[layer_idx]
        
#         # Apply orthogonalization to attention module's weight
#         if hasattr(attn_module, 'weight'):
#             original_weight = attn_module.weight.data
#             attn_module.weight.data = ortho_mod_fn(original_weight)
#             print(f"  Layer {layer_idx}: Orthogonalized attention weight")
        
#         # Apply orthogonalization to MLP module's weight
#         if hasattr(mlp_module, 'weight'):
#             original_weight = mlp_module.weight.data
#             mlp_module.weight.data = ortho_mod_fn(original_weight)
#             print(f"  Layer {layer_idx}: Orthogonalized MLP weight")
    
#     # Save the modified model
#     output_dir = os.path.join(cfg.artifact_path(), 'abliterated_model')
#     os.makedirs(output_dir, exist_ok=True)
    
#     print(f"\nSaving abliterated model to {output_dir}...")
#     model_base.model.save_pretrained(output_dir, safe_serialization=True)
#     model_base.tokenizer.save_pretrained(output_dir)
    
#     # Save metadata
#     metadata = {
#         "layers_modified": list(layers_to_modify),
#         "model_path": cfg.model_path,
#     }
#     with open(os.path.join(output_dir, 'abliteration_metadata.json'), 'w') as f:
#         json.dump(metadata, f, indent=4)
    
#     print(f"Model saved successfully to {output_dir}")
#     return output_dir

def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs

def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction")
    )

    with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)

def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_pipeline(model_path):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    model_base = construct_model_base(cfg.model_path)
    # TEST refusal_toks
    print(f"\n{'='*80}")
    print("TESTING REFUSAL TOKENS")
    print(f"{'='*80}")
    print(f"refusal_toks: {model_base.refusal_toks}")
    print(f"refusal_toks type: {type(model_base.refusal_toks)}")
    if hasattr(model_base.refusal_toks, 'shape'):
        print(f"refusal_toks shape: {model_base.refusal_toks.shape}")
    if len(model_base.refusal_toks) > 0:
        print(f"Example tokens decoded: {model_base.tokenizer.decode(model_base.refusal_toks[:5])}")
    print(f"vocab_size: {model_base.model.config.vocab_size}")
    print(f"{'='*80}\n")    
    # print(cfg.model_path)
    # print(model_base)
    # exit(-1)

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    # Filter datasets based on refusal scores
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)

    # 1. Generate candidate refusal directions
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    
    # 2. Select the most effective refusal direction
    pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    # 3a. Generate and save completions on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', dataset_name)

    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
    
    # 4a. Generate and save completions on harmless evaluation dataset
    harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless', dataset=harmless_test)
    
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, 'actadd', 'harmless', dataset=harmless_test)

    # 4b. Evaluate completions and save results on harmless evaluation dataset
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless', eval_methodologies=cfg.refusal_eval_methodologies)

    # 5. Evaluate loss on harmless datasets
    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline')
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation')
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd')

    # 6. Permanently modify and save the abliterated model
    print("\n" + "="*80)
    print("CREATING ABLITERATED MODEL WITH PERMANENT WEIGHT MODIFICATIONS")
    print("="*80)
    abliterated_model_path = orthogonalize_and_save_model(cfg, model_base, direction)
    print(f"\nAbliterated model saved to: {abliterated_model_path}")

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)