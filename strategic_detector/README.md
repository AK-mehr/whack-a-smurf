# strategic_detector

Implements the detector-design logic of Yang, Yao & Zhou (2026, *Management Science*,
"Strategic Disinformation Generation and Detection") on top of `SAML-D.csv`, translated
from their sender/receiver/detector game into AML transaction monitoring:

| paper                                      | this project                                                                                                                        |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| sender's type H / L                        | transaction is legitimate / is laundering                                                                                           |
| message m_H ("I'm legitimate")             | every laundering transaction tries to look legitimate (no AML analog of self-flagging, so only the m_H branch of the model applies) |
| classifier signal s                        | `Amount` (Control A) or distinct-receiver-count in a window (Control B)                                                             |
| alarm rule / detector {beta, alpha}        | a threshold on that signal; beta = Pr(flag \| laundering), alpha = Pr(flag \| legitimate)                                           |
| designer                                   | the compliance team choosing the threshold                                                                                          |
| lying cost C                               | cost to a launderer of restructuring to mimic a legitimate pattern                                                                  |
| Delta^S_L, Delta^R_H, Delta^R_L, Delta^S_H | payoffs you have to choose (see `scenario.ILLUSTRATIVE_PARAMS`)                                                                     |

## Structure

```
strategic_detector/
    config.py       EconomicParams (the paper's payoff parameters) + paths
    data.py         loads the notebook's cached parquet; ports distinct_receivers()
    detectors.py    roc_curve_from_score() -- full ROC curve for a control, not just a few budgets
    equilibrium.py  sigma_s_star(), equilibrium_payoffs() -- closed-form from Proposition 1 / Table A.1
    objective.py    designer_objective(), optimal_point() -- Lemma 6 / Proposition 3
    reporting.py    matplotlib plotting helpers
    scenario.py     ILLUSTRATIVE_PARAMS and the ROC-curve file paths shared by the scripts
scripts/
    build_roc_curves.py    Step 1-2: build & save each control's full ROC curve
    equilibrium_sweep.py   Step 4-5: sigma_S*(beta) along each ROC curve -- look for the Figure-3 hump
    optimal_detector.py    Step 6-7: beta* that maximizes the designer's objective, + its threshold
tests/
    test_detectors.py, test_equilibrium.py
```

## Running it

```
pip install -r requirements-dev.txt
python scripts/build_roc_curves.py
python scripts/equilibrium_sweep.py
python scripts/optimal_detector.py
pytest tests/
```

Run from the repo root (the scripts add it to `sys.path` themselves). The first script
needs `saml-d.parquet` to already exist (it's the notebook's own cache).

## Before trusting any numbers out of this

`scenario.ILLUSTRATIVE_PARAMS` are placeholders, not calibrated figures. `delta_r_low`
(cost of a missed laundering transaction) and `delta_s_low` (benefit to the launderer)
in particular should come from your own risk/compliance assumptions, not from this repo.
`EconomicParams` validates the paper's structural requirements (0 < C < min(Delta^S_H,
Delta^S_L)) and warns if `delta_s_low > delta_r_low`, which breaks the paper's welfare
assumption behind Propositions 3-5.

Also note: `equilibrium.py` and `objective.py` restrict every formula to the region
`0 < alpha < beta`. The paper only characterizes the semiseparating equilibrium there;
outside it (alpha = 0, alpha = beta, or alpha > beta) a different proposition applies,
and `optimal_point()` filters those points out of the ROC curve before searching.
