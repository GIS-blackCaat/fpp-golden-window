#!/usr/bin/env python3
"""FPP V4 Health Dashboard — radar chart + bar chart for model health check"""
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# Use English labels to avoid CJK font issues
DIMS_LABEL = ['GS\n(Structure)', 'MI\n(Info Flow)', 'Phase\n(Layer Diff.)', 'Deception\n(Dishonesty)', 'IPR\n(Info Loc.)']
HEALTH_LABELS = {'WEAK': 'Weak', 'MID': 'Medium', 'STRONG': 'Strong'}
import numpy as np

FAMILY_COLORS = {'SwiGLU': '#2196F3', 'GeGLU': '#FF9800', 'ReLU': '#4CAF50', 'GELU': '#9C27B0'}

def load_report(path):
    with open(path) as f: return json.load(f)

def plot_health_radar(report, out_path='fpp_health_radar.png'):
    """Radar chart: 6 FPP dimensions vs family healthy range."""
    fpp = report['fpp_baseline']
    family = report['architecture']['family']
    fam_color = FAMILY_COLORS.get(family, '#666')
    model_name = os.path.basename(report['model'].rstrip('/'))

    # Normalize each dimension to [0,1] using family-specific ranges
    # These ranges come from the 12-architecture study
    ranges = {
        'SwiGLU': {'GS':(0.70,0.95), 'MI':(0.0,0.12), 'Phase':(0.30,0.55), 'DC':(0.30,0.0), 'IPR':(0.0,0.10)},
        'GeGLU':  {'GS':(0.15,0.40), 'MI':(0.03,0.18), 'Phase':(0.20,0.45), 'DC':(0.30,0.0), 'IPR':(0.0,0.10)},
        'ReLU':   {'GS':(0.40,0.75), 'MI':(0.05,0.25), 'Phase':(0.15,0.35), 'DC':(0.30,0.0), 'IPR':(0.0,0.10)},
        'GELU':   {'GS':(0.35,0.65), 'MI':(0.0,0.03),  'Phase':(0.18,0.40), 'DC':(0.30,0.0), 'IPR':(0.0,0.10)},
    }
    r = ranges.get(family, ranges['SwiGLU'])

    dims = ['GS', 'MI', 'Phase', 'DC', 'IPR']
    dims_label = ['GS\n(结构有序)', 'MI\n(信息通道)', 'Phase\n(层分化)', 'Deception\n(欺骗度↓)', 'IPR\n(信息定位)']

    values = [fpp['gs'], fpp['mi'], fpp['ph'], fpp['dc'], fpp['ipr']]
    mins = [r['GS'][0], r['MI'][0], r['Phase'][0], r['DC'][0], r['IPR'][0]]
    maxs = [r['GS'][1], r['MI'][1], r['Phase'][1], r['DC'][1], r['IPR'][1]]

    # For DC, lower is better, so invert: normalized = 1 - (val-min)/(max-min)
    normalized = []
    for v, mn, mx, d in zip(values, mins, maxs, dims):
        if d == 'DC':
            norm = max(0, min(1, 1 - (v - mn) / (mx - mn + 1e-8)))
        else:
            norm = max(0, min(1, (v - mn) / (mx - mn + 1e-8)))
        normalized.append(norm)

    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist()
    normalized += normalized[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')

    ax.set_facecolor('#16213e')
    ax.fill(angles, [0.5]*len(angles), alpha=0.08, color='white')
    ax.fill(angles, normalized, alpha=0.35, color=fam_color)
    ax.plot(angles, normalized, 'o-', linewidth=2, color=fam_color, markersize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMS_LABEL, size=11, color='white')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['Weak', 'Medium', 'Strong'], size=9, color='#aaa')
    ax.set_title(f'FPP V4 Health: {model_name} ({family})', size=14, color='white', pad=20)
    ax.grid(True, alpha=0.3, color='white')

    status = report['health']['status']
    status_colors = {'🟢 HEALTHY': '#4CAF50', '🟡 NEEDS ATTENTION': '#FF9800', '🔴 AT RISK': '#f44336'}
    fig.text(0.5, 0.02, f'Status: {status}', ha='center', size=14,
             color=status_colors.get(status, 'white'), weight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"  Radar chart saved: {out_path}")

def plot_beta_comparison(reports_data, out_path='fpp_beta_comparison.png'):
    """Bar chart: GS and MI across different models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    models = []
    gs_vals, mi_vals, colors = [], [], []
    for data in reports_data:
        report = data['report']
        fpp = report['fpp_baseline']
        family = report['architecture']['family']
        name = os.path.basename(report['model'].rstrip('/'))[:15]
        models.append(name)
        gs_vals.append(fpp['gs'])
        mi_vals.append(fpp['mi'])
        colors.append(FAMILY_COLORS.get(family, '#666'))

    # GS bar chart
    ax = axes[0]
    ax.bar(range(len(models)), gs_vals, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    for i, (v, m) in enumerate(zip(gs_vals, models)):
        ax.text(i, v+0.01, f'{v:.3f}', ha='center', color='white', size=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, size=9, color='white', rotation=30)
    ax.set_title('GS (结构有序度)', size=12, color='white')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.grid(axis='y', alpha=0.2, color='white')

    # MI bar chart
    ax = axes[1]
    ax.bar(range(len(models)), mi_vals, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    for i, (v, m) in enumerate(zip(mi_vals, models)):
        ax.text(i, v+0.01, f'{v:.3f}', ha='center', color='white', size=9)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, size=9, color='white', rotation=30)
    ax.set_title('MI (信息通道容量)', size=12, color='white')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.grid(axis='y', alpha=0.2, color='white')

    fig.suptitle('FPP V4 Cross-Architecture Comparison', size=14, color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"  Bar chart saved: {out_path}")

if __name__ == '__main__':
    # Look for health reports
    reports_dir = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(reports_dir) if f.startswith('fpp_health_') and f.endswith('.json')]
    if not files:
        print("No fpp_health_*.json files found. Run fpp_health_check.py first.")
        sys.exit(1)

    reports_data = []
    for fn in sorted(files):
        path = os.path.join(reports_dir, fn)
        report = load_report(path)
        reports_data.append({'report': report, 'path': path})

    # Individual radar charts
    for data in reports_data:
        report = data['report']
        name = os.path.basename(data['path']).replace('.json','')
        plot_health_radar(report, os.path.join(reports_dir, f'{name}.png'))

    # Cross-architecture comparison bar chart
    if len(reports_data) > 1:
        plot_beta_comparison(reports_data, os.path.join(reports_dir, 'fpp_cross_arch_comparison.png'))

    print(f"\n  Generated {len(reports_data)} radar charts + 1 comparison chart")
