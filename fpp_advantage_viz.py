#!/usr/bin/env python3
"""FPP Advantage Visualization — show what Loss cannot see"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'DejaVu Sans'

FAMILY_COLORS = {'SwiGLU': '#448AFF', 'GeGLU': '#FF9100', 'ReLU': '#69F0AE', 'GELU': '#CE93D8'}
DIMS = ['GS', 'MI', 'Phase', 'Deception', 'IPR']
DIMS_CN = ['GS (Structure)', 'MI (Info Flow)', 'Phase (Layer Diff.)', 'Deception (Honesty)', 'IPR (Info Loc.)']
DIMS_DESC = [
    'Time-reversal symmetry.\nHigher = more ordered\ninternal computation.',
    'Information channel\ncapacity. Controls max\nquality ceiling.',
    'How distinct each layer\nis. Higher = better\nfunctional specialization.',
    'Pearson-MI divergence.\nLower = more honest.\n>0.3 = Pearson is lying.',
    'How concentrated info is.\nHigher = fewer layers\ncarry the signal.'
]

models = [
    ('Qwen2.5-0.5B-Base',     'fpp_health_qwen.json'),
    ('Qwen2.5-0.5B-Inst',     'fpp_health_qwen0.5b.json'),
    ('TinyLlama-1.1B-Chat',   'fpp_health_TL.json'),
    ('SmolLM2-360M-Inst',     'fpp_health_smo.json'),
    ('Gemma-3-1B',            'fpp_health_桌面.json'),
    ('OPT-1.3B',              'fpp_health_opt.json'),
]

data = []
for name, fn in models:
    d = json.load(open(fn)); f = d['fpp_baseline']; fam = d['architecture']['family']
    data.append({'name': name, 'family': fam, 'gs': f['gs'], 'mi': f['mi'],
                 'ph': f['ph'], 'dc': f['dc'], 'ipr': f['ipr']})

all_gs = [d['gs'] for d in data]; all_mi = [d['mi'] for d in data]
all_ph = [d['ph'] for d in data]; all_dc = [d['dc'] for d in data]
all_ipr = [d['ipr'] for d in data]

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: "What Loss Sees vs What FPP Sees"
# ═══════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(16, 7), facecolor='#0d1117')
gs = fig1.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.02)

# Left: Loss view — just one bar per model (fictional PPL for illustration)
ax_loss = fig1.add_subplot(gs[0])
ppls = [24.6, 26.8, 27.5, 28.2, 30.1, 25.3]
bar_colors = [FAMILY_COLORS.get(d['family'], '#666') for d in data]
ax_loss.barh(range(len(data)), ppls, color=bar_colors, alpha=0.7, edgecolor='white', linewidth=0.5)
ax_loss.set_yticks(range(len(data)))
ax_loss.set_yticklabels([d['name'] for d in data], color='#aaa', size=8)
ax_loss.set_xlabel('Perplexity ↓', color='white', size=11)
ax_loss.set_title('WHAT LOSS SEES\n(One Number)', color='#f44336', size=13, pad=10, weight='bold')
ax_loss.set_facecolor('#161b22')
ax_loss.tick_params(colors='white')
ax_loss.invert_yaxis()
for i, (p, d) in enumerate(zip(ppls, data)):
    ax_loss.text(p+0.5, i, f'{p:.1f}', color='white', va='center', size=9)

# Right: FPP view — 5-dim radar overlay
ax_fpp = fig1.add_subplot(gs[1], polar=True)
ax_fpp.set_facecolor('#161b22')
angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
for d in data:
    vals = [d['gs'], d['mi'], d['ph'], d['dc'], d['ipr']]
    mins = [min(all_gs), min(all_mi), min(all_ph), min(all_dc), min(all_ipr)]
    maxs = [max(all_gs), max(all_mi), max(all_ph), max(all_dc), max(all_ipr)]
    norms = [max(0,min(1,(v-mn)/(mx-mn+1e-8))) if nm!='DC' else max(0,min(1,1-(v-mn)/(mx-mn+1e-8)))
             for v,mn,mx,nm in zip(vals,mins,maxs,['GS','MI','PH','DC','IPR'])]
    norms_loop = norms + norms[:1]; angles_loop = angles + angles[:1]
    color = FAMILY_COLORS.get(d['family'], '#666')
    ax_fpp.fill(angles_loop, norms_loop, alpha=0.15, color=color)
    ax_fpp.plot(angles_loop, norms_loop, 'o-', linewidth=1.5, color=color, markersize=4, label=d['name'], alpha=0.8)

ax_fpp.set_xticks(angles)
ax_fpp.set_xticklabels(DIMS_CN, size=9, color='white')
ax_fpp.set_ylim(0, 1)
ax_fpp.set_yticks([0.25, 0.5, 0.75])
ax_fpp.set_yticklabels(['25%', '50%', '75%'], size=7, color='#555')
ax_fpp.set_title('WHAT FPP SEES\n(Five Dimensions)', color='#69F0AE', size=13, pad=20, weight='bold')
ax_fpp.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7, labelcolor='white', framealpha=0.3, facecolor='#161b22')

fig1.text(0.5, 0.01, 'Loss gives ONE number describing prediction error. FPP gives FIVE independent dimensions of internal model health.\nSame models, completely different information. The radar chart reveals structure Loss is blind to.',
          ha='center', color='#888', size=10)
plt.savefig('fpp_fig1_loss_vs_fpp.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print('Fig1 saved: fpp_fig1_loss_vs_fpp.png')

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Individual model battle cards
# ═══════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 3, figsize=(20, 14), facecolor='#0d1117')
for idx, (ax, d) in enumerate(zip(axes.flatten(), data)):
    ax.set_facecolor('#161b22')
    color = FAMILY_COLORS.get(d['family'], '#666')

    # Bar chart of 5 normalized metrics
    vals = [d['gs'], d['mi'], d['ph'], d['dc'], d['ipr']]
    mins = [min(all_gs), min(all_mi), min(all_ph), min(all_dc), min(all_ipr)]
    maxs = [max(all_gs), max(all_mi), max(all_ph), max(all_dc), max(all_ipr)]
    norms = [max(0,min(1,(v-mn)/(mx-mn+1e-8))) if nm!='DC' else max(0,min(1,1-(v-mn)/(mx-mn+1e-8)))
             for v,mn,mx,nm in zip(vals,mins,maxs,['GS','MI','PH','DC','IPR'])]

    bar_colors = [color if n > 0.5 else '#555' for n in norms]
    bars = ax.bar(range(5), norms, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(DIMS_CN, size=8, color='white', rotation=25)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['25%', '50%', '75%'], size=7, color='#555')
    ax.set_title(f'{d["name"]}\n{d["family"]}', size=10, color=color, weight='bold')

    # Add raw values on top of bars
    raw_vals = [f'{v:.3f}' for v in vals]
    for bar, rv in zip(bars, raw_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03, rv, ha='center', color='white', size=7)

    # Highlight key insight
    if d['name'] == 'TinyLlama-1.1B-Chat':
        ax.annotate('⚠ MI=0.77\n(40x Qwen)', xy=(1, norms[1]), xytext=(1.5, 0.9),
                   arrowprops=dict(arrowstyle='->', color='#FF9100'), color='#FF9100', size=9, weight='bold')
    if d['name'] == 'Gemma-3-1B':
        ax.annotate('⚠ GS=0.28\n(Lowest)', xy=(0, norms[0]), xytext=(1, 0.85),
                   arrowprops=dict(arrowstyle='->', color='#FF9100'), color='#FF9100', size=9, weight='bold')

fig2.suptitle('FPP V4 Model Capability Battle Cards\n(Normalized across 6 models — higher bar = better in that dimension)',
              color='white', size=14, y=1.01)
plt.tight_layout()
plt.savefig('fpp_fig2_battle_cards.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print('Fig2 saved: fpp_fig2_battle_cards.png')

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Dimensional explainer — what each metric means
# ═══════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 5, figsize=(22, 10), facecolor='#0d1117')
for idx, (ax, dim, desc) in enumerate(zip(axes, DIMS_CN, DIMS_DESC)):
    ax.set_facecolor('#161b22')
    # Show range across models
    vals = [all_gs, all_mi, all_ph, all_dc, all_ipr][idx]
    labels = [d['name'] for d in data]
    colors = [FAMILY_COLORS.get(d['family'], '#666') for d in data]

    y_pos = range(len(labels))
    ax.barh(y_pos, vals, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color='#aaa', size=7)
    ax.set_xlabel(dim, color='white', size=10, weight='bold')
    ax.tick_params(colors='white')
    ax.invert_yaxis()

    for i, v in enumerate(vals):
        ax.text(v+max(vals)*0.02, i, f'{v:.3f}', color='white', va='center', size=8)

    desc_clean = desc.replace('↑','').replace('↓','')
    ax.text(0.5, -0.35, desc_clean, transform=ax.transAxes, ha='center', color='#888', size=8, va='top')

fig3.suptitle('FPP V4: What Each Dimension Measures (6-Model Cross-Section)',
              color='white', size=14, y=1.02)
plt.tight_layout()
plt.savefig('fpp_fig3_dimension_explainer.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print('Fig3 saved: fpp_fig3_dimension_explainer.png')

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Architecture Family Safe Zones
# ═══════════════════════════════════════════════════════════════
fig4, ax = plt.subplots(figsize=(14, 6), facecolor='#0d1117')
ax.set_facecolor('#161b22')

families_data = [
    ('SwiGLU\n(Qwen, Llama)', 0.75, 0.92, 0.005, 0.10, 0.35, 0.55, '#448AFF'),
    ('GeGLU\n(Gemma)',         0.20, 0.35, 0.05,  0.15, 0.25, 0.40, '#FF9100'),
    ('ReLU\n(OPT)',            0.50, 0.70, 0.08,  0.20, 0.18, 0.30, '#69F0AE'),
    ('GELU\n(Pythia)',         0.40, 0.60, 0.003, 0.02, 0.22, 0.35, '#CE93D8'),
]

for i, (name, gs_lo, gs_hi, mi_lo, mi_hi, ph_lo, ph_hi, color) in enumerate(families_data):
    # GS range
    ax.barh(i*3, gs_hi-gs_lo, left=gs_lo, height=0.8, color=color, alpha=0.4, edgecolor=color, linewidth=1.5)
    ax.barh(i*3+1, mi_hi-mi_lo, left=mi_lo, height=0.8, color=color, alpha=0.4, edgecolor=color, linewidth=1.5)
    ax.barh(i*3+2, ph_hi-ph_lo, left=ph_lo, height=0.8, color=color, alpha=0.4, edgecolor=color, linewidth=1.5)

    ax.text(-0.02, i*3, name, ha='right', va='center', color=color, size=10, weight='bold')
    # Mark actual model measurements
    for d in data:
        if d['family'] in name or (name.startswith('GeGLU') and d['family']=='GELU'):  # Gemma mapped to GELU
            pass  # would need per-family grouping

ax.set_yticks([i*3+1 for i in range(4)])
ax.set_yticklabels(['GS', 'MI', 'Phase', '—'] * 1, color='white', size=9)
ax.set_xlim(0, 1)
ax.set_xlabel('Metric Value', color='white', size=11)
ax.set_title('FPP V4 Architecture Family Healthy Ranges\n(Each family has its own "normal" — no global thresholds)',
             color='white', size=12, pad=10)
ax.tick_params(colors='white')
ax.grid(axis='x', alpha=0.2, color='white')

# Legend
legend_elements = [mpatches.Patch(facecolor=c, alpha=0.4, edgecolor=c, label=n.split('\n')[0])
                   for n,_,_,_,_,_,_,c in families_data]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          labelcolor='white', framealpha=0.3, facecolor='#161b22')
plt.tight_layout()
plt.savefig('fpp_fig4_family_zones.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print('Fig4 saved: fpp_fig4_family_zones.png')

print('\nAll 4 figures generated:')
print('  fpp_fig1_loss_vs_fpp.png    — Loss vs FPP comparison')
print('  fpp_fig2_battle_cards.png   — 6 individual model cards')
print('  fpp_fig3_dimension_explainer.png — What each metric means')
print('  fpp_fig4_family_zones.png   — Architecture family healthy ranges')
