"""Regenerate ALL poster figures at 400 DPI with large fonts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
import numpy as np

DPI = 400
out = 'C:/Users/chntw/Documents/7180/DD_v1/results/figures'

# ================================================================
# Fig 0: 3-Phase Framework
# ================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')

phases = [
    (0.3, 7.0, 4.0, 2.2, 'Phase 1: Baseline',
     '2\u00d72 factorial\nClaude / Qwen\n\u00d7 vision / vision+text\n271 Misviz charts', '#1F4E79'),
    (5.0, 7.0, 4.0, 2.2, 'Phase 2: Pipeline',
     '10 architectures\nOCR, DePlot, ViT\nas post-processors\nvs. prompt injection', '#2E7D32'),
    (9.7, 7.0, 4.0, 2.2, 'Phase 3: Validation',
     'Best pipeline on\n13 SEC 10-K filings\nvs. comment letter\nground truth', '#755882'),
]
for x, y, w, h, title, sub, color in phases:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h-0.4, title, ha='center', va='center',
            fontsize=18, color='white', fontweight='bold')
    ax.text(x+w/2, y+h/2-0.3, sub, ha='center', va='center',
            fontsize=15, color='#E0E0E0', linespacing=1.4)

for i in range(2):
    x1 = phases[i][0]+phases[i][2]+0.1; x2 = phases[i+1][0]-0.1
    y = phases[i][1]+phases[i][3]/2
    ax.annotate('', xy=(x2,y), xytext=(x1,y),
                arrowprops=dict(arrowstyle='->', color='#333', lw=3))

outputs = [
    (2.3, 5.5, 'Binary F1 = 83%\nbut PT F1 = 39.3%', '#C62828'),
    (7.0, 5.5, 'Best: CLEAN Veto\n+ Classifier = 45.2%', '#2E7D32'),
    (11.7, 5.5, 'Precision = 1.0\nRecall = 37.5%', '#755882'),
]
for x, y, txt, color in outputs:
    rect = FancyBboxPatch((x-1.5, y-0.7), 3.0, 1.4, boxstyle="round,pad=0.1",
                          facecolor='white', edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, txt, ha='center', va='center', fontsize=14, color=color, fontweight='bold')
    ax.annotate('', xy=(x, y+0.7), xytext=(x, 7.0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

ax.text(2.3, 4.3, 'Motivates RQ3', ha='center', fontsize=14, color='#C62828', fontstyle='italic')
ax.annotate('', xy=(5.0,4.5), xytext=(3.5,4.5),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=2, linestyle='--'))

rect_m = FancyBboxPatch((0.3,0.3), 13.4, 1.2, boxstyle="round,pad=0.1",
                        facecolor='#F5F5F5', edgecolor='#CCC', linewidth=1.5)
ax.add_patch(rect_m)
ax.text(7.0, 1.1, 'Evaluation Metrics', ha='center', fontsize=16, fontweight='bold', color='#1F4E79')
ax.text(3.5, 0.6, 'Binary F1: is chart misleading?', ha='center', fontsize=14, color='#555')
ax.text(7.0, 0.6, 'Per-Type F1: correct type across 12', ha='center', fontsize=14, color='#555')
ax.text(11.0, 0.6, 'Exact Match: full type set', ha='center', fontsize=14, color='#555')
ax.plot([5.2,5.2],[0.4,1.4], color='#CCC', lw=1)
ax.plot([8.8,8.8],[0.4,1.4], color='#CCC', lw=1)

plt.tight_layout()
plt.savefig(f'{out}/fig0_framework.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(); print('fig0 OK')

# ================================================================
# Fig 7: Pipeline Architecture
# ================================================================
fig, ax = plt.subplots(figsize=(16, 4.5))
ax.set_xlim(-0.2, 15); ax.set_ylim(-0.5, 5); ax.axis('off')

boxes = [
    (0.0,1.5,2.4,1.4,'Chart\nImage','#5B9BD5'),
    (3.1,1.5,2.4,1.4,'VLM\nInference','#1F4E79'),
    (6.2,1.5,2.4,1.4,'OCR\nCLEAN Veto','#2E7D32'),
    (9.3,1.5,2.6,1.4,'ViT Classifier\nSelective Veto','#2E7D32'),
    (12.5,1.5,2.4,1.4,'Final\nPrediction','#755882'),
]
for x,y,w,h,label,color in boxes:
    rect = FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white', linewidth=3)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center',
            fontsize=20, color='white', fontweight='bold')

for i in range(len(boxes)-1):
    x1=boxes[i][0]+boxes[i][2]+0.1; x2=boxes[i+1][0]-0.1; y=boxes[i][1]+boxes[i][3]/2
    ax.annotate('', xy=(x2,y), xytext=(x1,y),
                arrowprops=dict(arrowstyle='->', color='#333', lw=3))

ax.text(8.45, 0.5, 'Post-processing verifiers (effective)',
        ha='center', fontsize=19, color='#2E7D32', fontweight='bold', fontstyle='italic')
ax.annotate('', xy=(3.8,3.8), xytext=(1.0,3.8),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.5, linestyle='--'))
ax.text(0.5, 4.2, 'OCR / DePlot\ndata', ha='center', fontsize=16, color='#C62828')
ax.text(2.4, 4.3, 'Tool Injection (harmful)', ha='center', fontsize=19, color='#C62828', fontweight='bold')
ax.text(4.5, 3.8, '\u2717', ha='center', fontsize=30, color='#C62828', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{out}/fig7_pipeline_diagram.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(); print('fig7 OK')

# ================================================================
# Fig 8: Attention Dilution
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

types = ['3D', 'Dual Axis', 'Trunc. Axis', 'Pie Chart', 'Line Chart',
         'Misrep.', 'Inverted', 'Item Order', 'Axis Range',
         'Binning', 'Tick Intervals', 'Discretized']
recall_gen = [80, 73, 53, 60, 47, 42, 43, 15, 21, 10, 0, 0]
recall_ind = [80, 73, 53, 60, 47, 42, 43, 15, 21, 35, 21, 0]

colors_gen = ['#2E7D32' if r > 30 else '#C62828' if r == 0 else '#FF8F00' for r in recall_gen]
colors_ind = ['#2E7D32' if r > 30 else '#C62828' if r == 0 else '#FF8F00' for r in recall_ind]

ax1.barh(range(len(types)), recall_gen, color=colors_gen, edgecolor='white', height=0.7)
ax1.set_yticks(range(len(types))); ax1.set_yticklabels(types, fontsize=15)
ax1.set_xlabel('Recall (%)', fontsize=17); ax1.set_xlim(0, 100); ax1.invert_yaxis()
ax1.set_title('12-Type Prompt\n(Attention Diluted)', fontsize=20, fontweight='bold', color='#C62828')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=14)

ax2.barh(range(len(types)), recall_ind, color=colors_ind, edgecolor='white', height=0.7)
ax2.set_yticks(range(len(types))); ax2.set_yticklabels(types, fontsize=15)
ax2.set_xlabel('Recall (%)', fontsize=17); ax2.set_xlim(0, 100); ax2.invert_yaxis()
ax2.set_title('Individual Question\n(Focused)', fontsize=20, fontweight='bold', color='#2E7D32')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=14)

for i in [9, 10]:
    ax2.barh(i, recall_ind[i], color='#2E7D32', edgecolor='#1B5E20', linewidth=2.5, height=0.7)
    ax2.text(recall_ind[i]+2, i, f'+{recall_ind[i]-recall_gen[i]}',
             va='center', fontsize=15, fontweight='bold', color='#2E7D32')

fig.suptitle('Attention Dilution: VLM Misses Subtle Types in Multi-Type Prompts',
              fontsize=22, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{out}/fig8_attention_dilution.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(); print('fig8 OK')

# ================================================================
# Fig 5: Pipeline Bar Chart
# ================================================================
archs = ['VLM-only\nBaseline', 'OCR-\nInjected', 'DePlot-\nInjected',
         'CLEAN\nVeto', 'Sequential\nRe-ask', 'Self-\nConsistency',
         'CLEAN Veto\n+ Classifier']
pt_f1 = [39.3, 38.9, 37.6, 43.5, 40.1, 38.9, 45.2]
colors = ['#5B9BD5', '#C62828', '#C62828', '#2E7D32', '#1F4E79', '#999999', '#2E7D32']

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(range(len(archs)), pt_f1, color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, pt_f1):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(archs))); ax.set_xticklabels(archs, fontsize=13)
ax.set_ylabel('Per-Type F1 (%)', fontsize=16)
ax.set_title('Tool-Augmented Pipeline Comparison', fontsize=20, fontweight='bold')
ax.set_ylim(0, 52)
ax.axhline(y=39.3, color='#5B9BD5', linestyle='--', alpha=0.5)
ax.text(6.5, 39.8, 'Baseline', color='#5B9BD5', fontsize=12, alpha=0.7)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#C62828', label='Tool Injection (harmful)'),
    Patch(facecolor='#2E7D32', label='Post-processing (effective)'),
    Patch(facecolor='#5B9BD5', label='Baseline'),
    Patch(facecolor='#999999', label='No effect'),
], loc='upper left', fontsize=13)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=13)
plt.tight_layout()
plt.savefig(f'{out}/fig5_t2_pipeline_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(); print('fig5 OK')

# ================================================================
# Fig 9: Results Table
# ================================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')

columns = ['Architecture', 'PT F1', 'PT Prec', 'Binary F1', 'FP', 'VLM calls']
data = [
    ['VLM-only Baseline',     '39.3%', '35.7%', '83.0%', '157', '1'],
    ['OCR-Injected',          '38.9%', '31.3%', '80.1%', '224', '1'],
    ['DePlot-Injected',       '37.6%', '30.0%', '79.1%', '233', '1'],
    ['CLEAN Veto',            '43.5%', '44.3%', '77.0%', '107', '1'],
    ['Sequential Re-ask',     '40.1%', '30.1%', '80.5%', '276', '~7'],
    ['Self-Consistency (3x)', '38.9%', '29.1%', '80.4%', '285', '~19'],
    ['CLEAN Veto + Clf \u2605', '45.2%', '50.0%', '77.0%', '82', '1'],
]
row_colors = ['#E3F2FD','#FFEBEE','#FFEBEE','#E8F5E9','#E3F2FD','#F5F5F5','#C8E6C9']

table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center',
                 colColours=['#1F4E79']*len(columns))
table.auto_set_font_size(False); table.set_fontsize(15); table.scale(1, 2.0)

for j in range(len(columns)):
    cell = table[0, j]
    cell.set_text_props(color='white', fontweight='bold', fontsize=16)
    cell.set_facecolor('#1F4E79'); cell.set_edgecolor('white')

for i in range(len(data)):
    for j in range(len(columns)):
        cell = table[i+1, j]
        cell.set_facecolor(row_colors[i]); cell.set_edgecolor('white')
        cell.set_text_props(fontsize=15)
        if i == len(data)-1:
            cell.set_text_props(fontweight='bold', fontsize=16, color='#2E7D32')
        if i in [1, 2]:
            cell.set_text_props(color='#C62828')

plt.title('Results Summary: All Pipeline Architectures',
          fontsize=20, fontweight='bold', color='#1F4E79', pad=20)
plt.tight_layout()
plt.savefig(f'{out}/fig9_results_table.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(); print('fig9 OK')

# ================================================================
# Fig 1 (heatmap) — regenerate at high DPI
# ================================================================
import json
agg = json.load(open('C:/Users/chntw/Documents/7180/DD_v1/finchartaudit/results/aggregated_summary.json'))
labels = []
acc_vals, prec_vals, rec_vals, f1_vals = [], [], [], []
for entry in agg['rq1_rq2_misviz']:
    labels.append(entry['label'].strip().replace('   ', ' '))
    acc_vals.append(entry['accuracy'])
    prec_vals.append(entry['precision'])
    rec_vals.append(entry['recall'])
    f1_vals.append(entry['f1'])

data_matrix = np.array([acc_vals, prec_vals, rec_vals, f1_vals]).T
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(data_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, fontsize=18)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=16)

for i in range(len(labels)):
    for j in range(len(metrics)):
        val = data_matrix[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=18, fontweight='bold', color='white' if val > 0.5 else 'black')

ax.set_title('RQ1/RQ2 \u2014 Misviz Benchmark', fontsize=22, fontweight='bold', pad=15)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig(f'{out}/fig1_rq1_rq2_heatmap.png', dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(); print('fig1 heatmap OK')

print('\nAll figures regenerated at 400 DPI.')
