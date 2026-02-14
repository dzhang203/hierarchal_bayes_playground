"""
Comprehensive comparison of Stage 1 DiD estimators for staggered adoption.

Compares:
1. Borusyak (ImputationDiD) - Our implementation + diff-diff
2. Synthetic DiD (Arkhangelsky et al) - diff-diff
3. Callaway-Sant'Anna - diff-diff
4. Sun-Abraham - diff-diff

Focus: Finding the most RELIABLE first stage for HBM pipeline.
"""

import sys
sys.path.append('/Users/davidzhang/github/hierarchal_bayes_playground')
import pandas as pd
import numpy as np
from diff_diff import ImputationDiD, SyntheticDiD, CallawaySantAnna, SunAbraham

from src.data_generation_staggered_crossed import generate_staggered_adoption_data_crossed
from src.borusyak_estimator import borusyak_imputation_estimator, validate_borusyak_estimates
from src.borusyak_estimator_calibrated import borusyak_imputation_estimator_calibrated


def prepare_data_for_diffdiff(df, adoption_weeks):
    """
    Convert our data format to diff-diff format.

    diff-diff expects:
    - first_treat: cohort (adoption week), 0 for never-treated
    - treated: 0/1 indicator
    """
    df_dd = df.copy()

    df_dd['first_treat'] = df_dd['creator_id'].map(
        lambda x: adoption_weeks.get(x, 0)
    )

    # Create post-treatment indicator
    df_dd['post'] = (df_dd['treated'] == 1).astype(int)

    return df_dd


def compare_stage1_estimators(noise_scale=2.0, seed=42, verbose=True):
    """
    Compare all Stage 1 estimators on synthetic data with known ground truth.
    """

    print("="*80)
    print("STAGE 1 ESTIMATOR COMPARISON")
    print("="*80)
    print(f"Noise scale: {noise_scale}x (realistic revenue variance)")
    print()

    # =========================================================================
    # Generate data
    # =========================================================================

    if verbose:
        print("[0/6] Generating synthetic data with known ground truth...")
        print()

    df, truth = generate_staggered_adoption_data_crossed(
        n_genres=5,
        n_arpu_quintiles=5,
        n_creators_per_cell=8,
        noise_scale=noise_scale,
        seed=seed
    )

    true_att = np.mean([truth['creator_effects'][cid] for cid in truth['adoption_weeks'].keys()])

    if verbose:
        print(f"True ATT: {true_att:.4f}")
        print(f"True effect SD: {np.std(list(truth['creator_effects'].values())):.4f}")
        print()

    # Prepare data
    df_dd = prepare_data_for_diffdiff(df, truth['adoption_weeks'])

    results = {}

    # =========================================================================
    # 1. Our Borusyak Implementation (Calibrated)
    # =========================================================================

    if verbose:
        print("[1/6] Our Borusyak Implementation (calibrated SEs)...")

    ites_ours, _ = borusyak_imputation_estimator_calibrated(
        df,
        truth['never_treated'],
        calibration_factor=2.5,
        verbose=False
    )

    metrics_ours = validate_borusyak_estimates(
        ites_ours,
        truth['creator_effects'],
        verbose=False
    )

    results['Our Borusyak (calibrated)'] = {
        'att': ites_ours['effect_hat'].mean(),
        'att_se': ites_ours['se'].mean(),  # Avg individual SE
        'coverage': metrics_ours['coverage'],
        'rmse': metrics_ours['rmse'],
        'provides_ites': True,
        'ci_width': ites_ours['se'].mean() * 1.96 * 2
    }

    if verbose:
        print(f"  ATT: {results['Our Borusyak (calibrated)']['att']:.4f}")
        print(f"  Individual coverage: {results['Our Borusyak (calibrated)']['coverage']:.1%}")
        print(f"  RMSE: {results['Our Borusyak (calibrated)']['rmse']:.4f}")
        print()

    # =========================================================================
    # 2. diff-diff ImputationDiD (Borusyak)
    # =========================================================================

    if verbose:
        print("[2/6] diff-diff ImputationDiD (Borusyak)...")

    try:
        borusyak_dd = ImputationDiD(
            anticipation=0,
            n_bootstrap=0,  # Use analytical SEs
            cluster='creator_id'
        )

        borusyak_results = borusyak_dd.fit(
            data=df_dd,
            outcome='revenue',
            unit='creator_id',
            time='week',
            first_treat='first_treat',
            aggregate='overall'
        )

        att_covers = (true_att >= borusyak_results.overall_conf_int[0] and
                     true_att <= borusyak_results.overall_conf_int[1])

        results['ImputationDiD (diff-diff)'] = {
            'att': borusyak_results.overall_att,
            'att_se': borusyak_results.overall_se,
            'att_covers': att_covers,
            'provides_ites': False,  # Only aggregate
            'ci_width': borusyak_results.overall_conf_int[1] - borusyak_results.overall_conf_int[0]
        }

        if verbose:
            print(f"  ATT: {results['ImputationDiD (diff-diff)']['att']:.4f}")
            print(f"  SE: {results['ImputationDiD (diff-diff)']['att_se']:.4f}")
            print(f"  ATT covered: {'✓' if att_covers else '✗'}")
            print()

    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
            print()
        results['ImputationDiD (diff-diff)'] = None

    # =========================================================================
    # 3. SyntheticDiD (Arkhangelsky et al)
    # =========================================================================

    if verbose:
        print("[3/6] SyntheticDiD (Arkhangelsky et al)...")

    try:
        sdid = SyntheticDiD(
            n_bootstrap=0,  # Analytical SEs
            cluster='creator_id'
        )

        sdid_results = sdid.fit(
            data=df_dd,
            outcome='revenue',
            unit='creator_id',
            time='week',
            first_treat='first_treat',
            aggregate='overall'
        )

        att_covers = (true_att >= sdid_results.overall_conf_int[0] and
                     true_att <= sdid_results.overall_conf_int[1])

        results['SyntheticDiD'] = {
            'att': sdid_results.overall_att,
            'att_se': sdid_results.overall_se,
            'att_covers': att_covers,
            'provides_ites': False,
            'ci_width': sdid_results.overall_conf_int[1] - sdid_results.overall_conf_int[0]
        }

        if verbose:
            print(f"  ATT: {results['SyntheticDiD']['att']:.4f}")
            print(f"  SE: {results['SyntheticDiD']['att_se']:.4f}")
            print(f"  ATT covered: {'✓' if att_covers else '✗'}")
            print()

    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
            print()
        results['SyntheticDiD'] = None

    # =========================================================================
    # 4. Callaway-Sant'Anna
    # =========================================================================

    if verbose:
        print("[4/6] Callaway-Sant'Anna...")

    try:
        cs = CallawaySantAnna(
            control_group='never_treated',
            estimation_method='dr',  # Doubly robust
            n_bootstrap=0,
            cluster='creator_id'
        )

        cs_results = cs.fit(
            data=df_dd,
            outcome='revenue',
            unit='creator_id',
            time='week',
            first_treat='first_treat',
            aggregate='overall'
        )

        att_covers = (true_att >= cs_results.overall_conf_int[0] and
                     true_att <= cs_results.overall_conf_int[1])

        results['Callaway-SantAnna'] = {
            'att': cs_results.overall_att,
            'att_se': cs_results.overall_se,
            'att_covers': att_covers,
            'provides_ites': False,
            'ci_width': cs_results.overall_conf_int[1] - cs_results.overall_conf_int[0]
        }

        if verbose:
            print(f"  ATT: {results['Callaway-SantAnna']['att']:.4f}")
            print(f"  SE: {results['Callaway-SantAnna']['att_se']:.4f}")
            print(f"  ATT covered: {'✓' if att_covers else '✗'}")
            print()

    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
            print()
        results['Callaway-SantAnna'] = None

    # =========================================================================
    # 5. Sun-Abraham
    # =========================================================================

    if verbose:
        print("[5/6] Sun-Abraham...")

    try:
        sa = SunAbraham(
            n_bootstrap=0,
            cluster='creator_id'
        )

        sa_results = sa.fit(
            data=df_dd,
            outcome='revenue',
            unit='creator_id',
            time='week',
            first_treat='first_treat',
            aggregate='overall'
        )

        att_covers = (true_att >= sa_results.overall_conf_int[0] and
                     true_att <= sa_results.overall_conf_int[1])

        results['Sun-Abraham'] = {
            'att': sa_results.overall_att,
            'att_se': sa_results.overall_se,
            'att_covers': att_covers,
            'provides_ites': False,
            'ci_width': sa_results.overall_conf_int[1] - sa_results.overall_conf_int[0]
        }

        if verbose:
            print(f"  ATT: {results['Sun-Abraham']['att']:.4f}")
            print(f"  SE: {results['Sun-Abraham']['att_se']:.4f}")
            print(f"  ATT covered: {'✓' if att_covers else '✗'}")
            print()

    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
            print()
        results['Sun-Abraham'] = None

    # =========================================================================
    # 6. Our Borusyak (Original SEs - for comparison)
    # =========================================================================

    if verbose:
        print("[6/6] Our Borusyak (original SEs - baseline)...")

    ites_orig, _ = borusyak_imputation_estimator(
        df,
        truth['never_treated'],
        verbose=False
    )

    metrics_orig = validate_borusyak_estimates(
        ites_orig,
        truth['creator_effects'],
        verbose=False
    )

    results['Our Borusyak (original)'] = {
        'att': ites_orig['effect_hat'].mean(),
        'att_se': ites_orig['se'].mean(),
        'coverage': metrics_orig['coverage'],
        'rmse': metrics_orig['rmse'],
        'provides_ites': True,
        'ci_width': ites_orig['se'].mean() * 1.96 * 2
    }

    if verbose:
        print(f"  ATT: {results['Our Borusyak (original)']['att']:.4f}")
        print(f"  Individual coverage: {results['Our Borusyak (original)']['coverage']:.1%}")
        print()

    # =========================================================================
    # Summary comparison
    # =========================================================================

    print("="*80)
    print("COMPREHENSIVE COMPARISON")
    print("="*80)
    print(f"\nTrue ATT: {true_att:.4f}")
    print()

    # Table
    print(f"{'Method':<30} {'ATT':<10} {'SE/AvgSE':<12} {'Coverage':<15} {'CI Width':<12} {'ITEs?'}")
    print("-"*95)

    for method, res in results.items():
        if res is None:
            continue

        att_str = f"{res['att']:.4f}"
        se_str = f"{res['att_se']:.4f}"

        if 'coverage' in res:
            cov_str = f"{res['coverage']:.1%} (indiv)"
        elif 'att_covers' in res:
            cov_str = "✓" if res['att_covers'] else "✗"
        else:
            cov_str = "N/A"

        width_str = f"{res['ci_width']:.4f}"
        ites_str = "Yes" if res['provides_ites'] else "No"

        print(f"{method:<30} {att_str:<10} {se_str:<12} {cov_str:<15} {width_str:<12} {ites_str}")

    print("="*80)
    print()

    # =========================================================================
    # Detailed Analysis
    # =========================================================================

    print("DETAILED ANALYSIS:")
    print()

    # Bias
    print("1. Bias (ATT estimate - True ATT):")
    for method, res in results.items():
        if res is None:
            continue
        bias = res['att'] - true_att
        print(f"   {method:<30} {bias:>8.4f}")
    print()

    # Standard errors (aggregate level)
    print("2. Standard Errors (aggregate ATT level):")
    agg_methods = {k: v for k, v in results.items() if v and not v['provides_ites']}
    if agg_methods:
        ses = {k: v['att_se'] for k, v in agg_methods.items()}
        min_se_method = min(ses, key=ses.get)
        max_se_method = max(ses, key=ses.get)

        for method, se in ses.items():
            marker = " ← Tightest" if method == min_se_method else ""
            marker = " ← Most conservative" if method == max_se_method else marker
            print(f"   {method:<30} {se:>8.4f}{marker}")
    print()

    # Coverage
    print("3. Coverage:")
    print("   Aggregate methods (ATT coverage):")
    for method, res in results.items():
        if res is None or 'att_covers' not in res:
            continue
        status = "✓ Covered" if res['att_covers'] else "✗ Missed"
        print(f"     {method:<28} {status}")

    print("\n   Individual methods (creator-level coverage):")
    for method, res in results.items():
        if res is None or 'coverage' not in res:
            continue
        cov = res['coverage']
        status = "✓ Good" if cov >= 0.93 else "⚠️ Poor"
        print(f"     {method:<28} {cov:.1%} {status}")
    print()

    # RMSE (for ITE methods)
    print("4. RMSE (individual treatment effects - only ITE methods):")
    for method, res in results.items():
        if res is None or not res['provides_ites']:
            continue
        print(f"   {method:<30} {res['rmse']:.4f}")
    print()

    # =========================================================================
    # Recommendations
    # =========================================================================

    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    print("For HBM Pipeline (need individual ITEs):")
    print()

    ite_methods = {k: v for k, v in results.items() if v and v['provides_ites']}

    if ite_methods:
        # Find best coverage
        best_coverage_method = max(ite_methods, key=lambda k: ite_methods[k]['coverage'])
        best_coverage = ite_methods[best_coverage_method]['coverage']

        print(f"  ✅ RECOMMENDED: {best_coverage_method}")
        print(f"     Coverage: {best_coverage:.1%}")
        print(f"     RMSE: {ite_methods[best_coverage_method]['rmse']:.4f}")
        print(f"     Provides individual ITEs for HBM: Yes")
        print()

        print("  Why this choice:")
        print("    • Achieves proper individual-level coverage (target: 95%)")
        print("    • Provides creator-level treatment effects")
        print("    • Integrates seamlessly with HBM Stage 2")
        print()

    print()
    print("For Aggregate Analysis Only:")
    print()

    agg_methods_valid = {k: v for k, v in results.items()
                        if v and not v['provides_ites'] and v.get('att_covers')}

    if agg_methods_valid:
        # Prefer tightest SEs among methods that cover
        best_agg = min(agg_methods_valid, key=lambda k: agg_methods_valid[k]['att_se'])

        print(f"  ✅ RECOMMENDED: {best_agg}")
        print(f"     ATT: {agg_methods_valid[best_agg]['att']:.4f}")
        print(f"     SE: {agg_methods_valid[best_agg]['att_se']:.4f}")
        print(f"     Covered true ATT: ✓")
        print()

        print("  Alternative (if need group-time effects):")
        if 'Callaway-SantAnna' in agg_methods_valid:
            print("    • Callaway-SantAnna: Provides detailed group-time decomposition")

    print()
    print("="*80)

    return results, true_att


if __name__ == "__main__":
    results, true_att = compare_stage1_estimators(
        noise_scale=2.0,
        seed=42,
        verbose=True
    )
