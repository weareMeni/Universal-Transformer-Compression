import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import datetime

from src.models import StandardLLM, UniversalLLM
from src.data_loader import get_dyck_loaders, get_recall_loaders, get_listops_loaders

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
    
    with torch.no_grad():
        for x, y in dataloader_or_list:
            x, y = x.to(device), y.to(device)
            logits, ponder_cost = model(x)
            
            loss = criterion(logits, y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * y.size(0)
            ponder_sum += ponder_cost.mean().item() * y.size(0) 
            total += y.size(0)
            
    return loss_sum / total, (100 * correct / total), (ponder_sum / total if total > 0 else 0)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    TASKS = ["RECALL", "LISTOPS", "DYCK"] 
    MODELS = ["STANDARD_COMPRESSOR", "UNIVERSAL_COMPRESSOR"] 
    
    DIM = 256
    GLOBAL_BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 128
    ACCUMULATION_STEPS = GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE
    SEQ_LEN = 40
    
    os.makedirs("results", exist_ok=True)
    master_results = []

    for task in TASKS:
        for model_name in MODELS:
            is_universal = (model_name == "UNIVERSAL_COMPRESSOR")
            print(f"\n" + "="*50)
            print(f">>> TASK: {task} | MODEL: {model_name} | LENGTH: {SEQ_LEN} <<<")
            
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

            # --- MODEL INSTANTIATION & PRE-RUN STATS ---
            if is_universal:
                model = UniversalLLM(vocab_size, num_classes, DIM, nhead=4, max_steps=12, core_layers=1).to(device)
                optimizer = setup_universal_optimizer(model)
                print(f"Architecture: 1 Core Layer (Recurrent), Max Steps: 12, Bottleneck: 1x{DIM}")
            else:
                model = StandardLLM(vocab_size, num_classes, DIM, nhead=4, num_layers=4).to(device)
                optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
                print(f"Architecture: 4 Static Layers, Bottleneck: 1x{DIM}")

            tot_params, train_params = count_parameters(model)
            print(f"Parameters: {tot_params:,} Total | {train_params:,} Trainable")
            print("="*50)

            criterion = nn.CrossEntropyLoss()
            
            # Grokfast setup
            grokfast_ema = {}
            grokfast_alpha, grokfast_lambda = 0.98, 1.0

            # Grab a static subset of training data for fast train evaluation
            train_eval_subset = []
            for i, batch in enumerate(trainloader):
                if i >= 10: break
                train_eval_subset.append(batch)

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

                if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                    _, t_acc, t_ponder = evaluate(model, testloader, criterion, device)
                    _, tr_acc, _ = evaluate(model, train_eval_subset, criterion, device)
                    
                    if is_universal:
                        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss_avg:.4f} | Train Acc: {tr_acc:.1f}% | Test Acc: {t_acc:.1f}% | Avg Ponder: {t_ponder:.2f} steps")
                    else:
                        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss_avg:.4f} | Train Acc: {tr_acc:.1f}% | Test Acc: {t_acc:.1f}%")
                else:
                    tr_acc, t_acc, t_ponder = 0.0, 0.0, 0.0 
                    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss_avg:.4f}")
                    
            master_results.append({
                "Task": task, 
                "Model": model_name, 
                "Train_Acc": tr_acc,
                "Test_Acc": t_acc, 
                "Avg_Ponder": t_ponder if is_universal else "N/A"
            })
            
    df = pd.DataFrame(master_results)
    print("\n" + "="*30 + "\nFINAL BENCHMARK\n" + "="*30)
    print(df.to_string(index=False))
    df.to_csv(f"results/compressor_benchmark_{datetime.datetime.now().strftime('%m%d_%H%M')}.csv", index=False)

if __name__ == "__main__":
    main()