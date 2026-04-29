import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

exp1_results = np.load(os.path.join(script_dir, 'exp1_results.npy'), allow_pickle=True).item()
exp2_results = np.load(os.path.join(script_dir, 'exp2_results.npy'), allow_pickle=True).item()
exp3_results = np.load(os.path.join(script_dir, 'exp3_results.npy'), allow_pickle=True).item()

epochs_exp1 = len(exp1_results['val_losses'])
epochs_exp2 = len(exp2_results['val_losses'])
epochs_exp3 = exp3_results['actual_epochs']

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
plt.plot(range(1, epochs_exp1 + 1), exp1_results['train_losses'], 'b-', linewidth=2, label='Exp1-SGD Train', alpha=0.7)
plt.plot(range(1, epochs_exp1 + 1), exp1_results['val_losses'], 'b--', linewidth=2, label='Exp1-SGD Val', alpha=0.7)
plt.plot(range(1, epochs_exp2 + 1), exp2_results['train_losses'], 'g-', linewidth=2, label='Exp2-Adam Train', alpha=0.7)
plt.plot(range(1, epochs_exp2 + 1), exp2_results['val_losses'], 'g--', linewidth=2, label='Exp2-Adam Val', alpha=0.7)
plt.plot(range(1, epochs_exp3 + 1), exp3_results['train_losses'], 'r-', linewidth=2, label='Exp3-Adam(BS128+ES) Train', alpha=0.7)
plt.plot(range(1, epochs_exp3 + 1), exp3_results['val_losses'], 'r--', linewidth=2, label='Exp3-Adam(BS128+ES) Val', alpha=0.7)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Experiment 1-3: Training and Validation Loss Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(1, max(epochs_exp1, epochs_exp2, epochs_exp3))

plt.subplot(2, 1, 2)
plt.plot(range(1, epochs_exp1 + 1), exp1_results['train_accuracies'], 'b-', linewidth=2, label='Exp1-SGD Train', alpha=0.7)
plt.plot(range(1, epochs_exp1 + 1), exp1_results['val_accuracies'], 'b--', linewidth=2, label='Exp1-SGD Val', alpha=0.7)
plt.plot(range(1, epochs_exp2 + 1), exp2_results['train_accuracies'], 'g-', linewidth=2, label='Exp2-Adam Train', alpha=0.7)
plt.plot(range(1, epochs_exp2 + 1), exp2_results['val_accuracies'], 'g--', linewidth=2, label='Exp2-Adam Val', alpha=0.7)
plt.plot(range(1, epochs_exp3 + 1), exp3_results['train_accuracies'], 'r-', linewidth=2, label='Exp3-Adam(BS128+ES) Train', alpha=0.7)
plt.plot(range(1, epochs_exp3 + 1), exp3_results['val_accuracies'], 'r--', linewidth=2, label='Exp3-Adam(BS128+ES) Val', alpha=0.7)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Experiment 1-3: Training and Validation Accuracy Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=9, loc='lower right')
plt.grid(True, alpha=0.3)
plt.xlim(1, max(epochs_exp1, epochs_exp2, epochs_exp3))
plt.ylim(80, 100)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'loss_curve_comparison.png'), dpi=300, bbox_inches='tight')
print("Comparison plot saved as 'loss_curve_comparison.png'")
plt.show()

print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"\nExp1 (SGD lr=0.01 BS=64):")
print(f"  Best Val Accuracy: {exp1_results['best_val_acc']:.2f}%")
print(f"  Final Val Accuracy: {exp1_results['final_val_acc']:.2f}%")
print(f"  Best Val Loss: {exp1_results['best_val_loss']:.4f}")
print(f"  Epochs: {epochs_exp1}")

print(f"\nExp2 (Adam lr=0.001 BS=64):")
print(f"  Best Val Accuracy: {exp2_results['best_val_acc']:.2f}%")
print(f"  Final Val Accuracy: {exp2_results['final_val_acc']:.2f}%")
print(f"  Best Val Loss: {exp2_results['best_val_loss']:.4f}")
print(f"  Epochs: {epochs_exp2}")

print(f"\nExp3 (Adam lr=0.001 BS=128 + Early Stopping):")
print(f"  Best Val Accuracy: {exp3_results['best_val_acc']:.2f}%")
print(f"  Best Val Loss: {exp3_results['best_val_loss']:.4f}")
print(f"  Actual Epochs: {epochs_exp3} (Early Stopped)")

print("\n" + "="*60)
