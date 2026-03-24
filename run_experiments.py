import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import datetime
import time

from src.models import StandardLLM, UniversalLLM
from src.data_loader import get_dyck_loaders, get_recall_loaders, get_listops_loaders

def measure_inference_vram(model, testloader, device):
    """Measures pure forward-pass VRAM without autograd/optimizer bloat."""
    model.eval()
    
    # 1. Force the allocator to clear out training graphs and optimizer states
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == 'mps':
        torch.mps.empty_cache()
        
    with torch.no_grad():
        x, _ = next(iter(testloader))
        x = x.to(device)
        
        # Run a single forward pass
        _ = model(x)
        
        # 2. Measure the clean footprint
        if device.type == 'cuda':
            return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        elif device.type == 'mps':
            try:
                # current_allocated_memory is much more accurate for inference footprint
                return torch.mps.current_allocated_memory() / (1024 ** 2)
            except AttributeError:
                return torch.mps.driver_allocated_memory() / (1024 ** 2)
        return 0.0

def estimate_mi_gaussian_proxy(z, y, eps=1e-2):
    if not isinstance(z, torch.Tensor):
        return 0.0

    if z.dim() == 3:
        z = z.reshape(-1, z.size(-1))
        y = y.reshape(-1)
        
    valid_mask = y >= 0
    z = z[valid_mask]
    y = y[valid_mask]
    
    if z.size(0) < 2:
        return 0.0

    z_centered = z - z.mean(dim=0)
    cov_z = (z_centered.T @ z_centered) / (z.size(0) - 1)
    cov_z += torch.eye(z.size(1), device=z.device) * eps 
    _, logdet_z = torch.linalg.slogdet(cov_z)
    h_z = 0.5 * logdet_z

    h_z_given_y = 0.0
    classes, counts = torch.unique(y, return_counts=True)
    
    for c, count in zip(classes, counts):
        if count > 1:
            z_c = z[y == c]
            z_c_centered = z_c - z_c.mean(dim=0)
            cov_z_c = (z_c_centered.T @ z_c_centered) / (count - 1)
            cov_z_c += torch.eye(z_c.size(1), device=z.device) * eps
            _, logdet_z_c = torch.linalg.slogdet(cov_z_c)
            
            p_c = count / z.size(0)
            h_z_given_y += p_c * (0.5 * logdet_z_c)
            
    mi_proxy = h_z - h_z_given_y
    return max(0.0, mi_proxy.item())

def setup_universal_optimizer(model, lr=3e-4, base_wd=0.1, gate_wd=0.5):
    gate_params = [p for n, p in model.named_parameters() if "halting_classifier" in n and p.requires_grad]
    base_params = [p for n, p in model.named_parameters() if "halting_classifier" not in n and p.requires_grad]
    return optim.AdamW([
        {'params': base_params, 'weight_decay': base_wd},
        {'params': gate_params, 'weight_decay': gate_wd} 
    ], lr=lr)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def evaluate(model, dataloader_or_list, criterion, device):
    model.eval()
    correct, total, loss_sum, ponder_sum = 0, 0, 0.0, 0.0
    mi_sum, batches = 0.0, 0
    
    bottleneck_features = []
    
    def extract_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (tuple, list)):
            for item in x:
                res = extract_tensor(item)
                if res is not None:
                    return res
        return None

    def hook(module, hook_input, hook_output):
        tensor_in = extract_tensor(hook_input)
        if tensor_in is not None:
            bottleneck_features.append(tensor_in.detach())
        
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            
    if last_linear is not None:
        handle = last_linear.register_forward_hook(hook)
    else:
        handle = None

    with torch.no_grad():
        for x, y in dataloader_or_list:
            x, y = x.to(device), y.to(device)
            bottleneck_features.clear()
            
            logits, ponder_cost = model(x)
            loss = criterion(logits, y)
            
            if logits.dim() == 3: 
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)
                valid_mask = y_flat >= 0
                correct += (logits_flat.argmax(1) == y_flat)[valid_mask].sum().item()
                total += valid_mask.sum().item()
            else:
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
                
            loss_sum += loss.item() * x.size(0)
            ponder_sum += ponder_cost.mean().item() * x.size(0) 
            
            if len(bottleneck_features) > 0:
                z = bottleneck_features[-1]
                if isinstance(z, torch.Tensor):
                    batch_mi = estimate_mi_gaussian_proxy(z, y)
                    mi_sum += batch_mi
                    batches += 1
                
    if handle is not None:
        handle.remove()
            
    avg_loss = loss_sum / total if total > 0 else 0.0
    avg_acc = (100 * correct / total) if total > 0 else 0.0
    avg_ponder = ponder_sum / total if total > 0 else 0.0
    avg_mi = mi_sum / batches if batches > 0 else 0.0
    
    return avg_loss, avg_acc, avg_ponder, avg_mi

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    TASKS = ["LISTOPS", "DYCK", "RECALL"] 
    MODELS = ["STANDARD_COMPRESSOR", "UNIVERSAL_COMPRESSOR"] 
    
    DIM = 256
    GLOBAL_BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 128
    ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE
    SEQ_LEN = 40
    
    os.makedirs("results", exist_ok=True)
    master_results = []

    for t_idx, task in enumerate(TASKS):
        if task == "RECALL":
            gate_decay = 0.05
            use_grokfast = False
            print(f"\n[NOTE] Task is RECALL. Disabling Grokfast and lowering gate_wd to {gate_decay}.", flush=True)
        else:
            gate_decay = 0.5
            use_grokfast = True
        
        for m_idx, model_name in enumerate(MODELS):
            # Clear memory strictly between model runs to avoid watermark bleed
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
                
            is_universal = (model_name == "UNIVERSAL_COMPRESSOR")
            print(f"\n" + "="*70, flush=True)
            print(f">>> Benchmark [{t_idx+1}/{len(TASKS)}]: TASK: {task} | MODEL: {model_name} <<<", flush=True)
            
            if task == "DYCK":
                trainloader, testloader, vocab_size, num_classes = get_dyck_loaders(MICRO_BATCH_SIZE, train_len=SEQ_LEN, test_len=SEQ_LEN)
                epochs = 20
            elif task == "RECALL":
                recall_len = min(SEQ_LEN, 10)
                trainloader, testloader, vocab_size, num_classes = get_recall_loaders(MICRO_BATCH_SIZE, train_len=recall_len, test_len=recall_len)
                epochs = 100
            elif task == "LISTOPS":
                trainloader, testloader, vocab_size, num_classes = get_listops_loaders(MICRO_BATCH_SIZE, train_len=SEQ_LEN, test_len=SEQ_LEN)
                epochs = 50

            if is_universal:
                model = UniversalLLM(vocab_size, num_classes, DIM, nhead=4, max_steps=12, core_layers=1).to(device)
                optimizer = setup_universal_optimizer(model, gate_wd=gate_decay)
            else:
                model = StandardLLM(vocab_size, num_classes, DIM, nhead=4, num_layers=4).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

            tot_params, train_params = count_parameters(model)
            print(f"Parameters: {tot_params:,} Total | {train_params:,} Trainable", flush=True)
            print("="*70, flush=True)

            criterion = nn.CrossEntropyLoss()
            grokfast_ema = {}
            grokfast_alpha, grokfast_lambda = 0.98, 1.0

            train_eval_subset = []
            for i, batch in enumerate(trainloader):
                if i >= 10: break
                train_eval_subset.append(batch)

            task_start_time = time.time()

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                optimizer.zero_grad()
                
                for step, (x, y) in enumerate(trainloader):
                    x, y = x.to(device), y.to(device)
                    logits, _ = model(x)
                    loss = criterion(logits, y) 

                    (loss / ACCUMULATION_STEPS).backward()

                    if (step + 1) % ACCUMULATION_STEPS == 0:
                        if use_grokfast:
                            with torch.no_grad():
                                for name, p in model.named_parameters():
                                    if p.grad is not None:
                                        if name not in grokfast_ema:
                                            grokfast_ema[name] = p.grad.clone().detach()
                                        else:
                                            grokfast_ema[name].mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                                        p.grad.add_(grokfast_ema[name], alpha=grokfast_lambda)

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item()

                train_loss_avg = epoch_loss / len(trainloader)

                _, t_acc, t_ponder, t_mi = evaluate(model, testloader, criterion, device)
                _, tr_acc, _, _ = evaluate(model, train_eval_subset, criterion, device)
                
                if is_universal:
                    print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss_avg:.4f} | Train Acc: {tr_acc:5.1f}% | Test Acc: {t_acc:5.1f}% | Test MI: {t_mi:6.2f} nats | Ponder: {t_ponder:4.2f} steps", flush=True)
                else:
                    print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss_avg:.4f} | Train Acc: {tr_acc:5.1f}% | Test Acc: {t_acc:5.1f}% | Test MI: {t_mi:6.2f} nats", flush=True)

            task_duration = time.time() - task_start_time
            
            # Use the new inference measurement here
            inf_vram = measure_inference_vram(model, testloader, device)
                    
            master_results.append({
                "Task": task, 
                "Model": model_name, 
                "Train_Acc": round(tr_acc, 2),
                "Test_Acc": round(t_acc, 2), 
                "Test_MI": round(t_mi, 3),
                "Avg_Ponder": round(t_ponder, 2) if is_universal else "N/A",
                "Inf_VRAM_MB": round(inf_vram, 2),
                "Total_Time_Sec": round(task_duration, 2)
            })
            
    df = pd.DataFrame(master_results)
    print("\n" + "="*30 + "\nFINAL BENCHMARK\n" + "="*30, flush=True)
    print(df.to_string(index=False), flush=True)
    
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    df.to_csv(f"results/compressor_benchmark_{timestamp}.csv", index=False)
    print(f"\nResults saved to results/compressor_benchmark_{timestamp}.csv", flush=True)

if __name__ == "__main__":
    main()