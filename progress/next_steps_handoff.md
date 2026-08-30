# Hedge Fund ML Engine - Project Handoff & Progress

## Current State & Recent Accomplishments
* **Phase 4 (Alternative Data) Implemented:** We successfully integrated daily FINRA RegSHO short volume data. We built the `finra_short_volume.py` scraper, integrated it into `run_ml_pipeline.py`, and created trailing z-score features (`sv_short_ratio_z21`, etc.).
* **European Short Data Investigation:** Explored alternative sources for EU short volume (ESMA, BaFin, Shortsells.nl). Concluded that true historical daily short volume isn't available for EU equities. Because of this, 46 EU tickers are dropped when testing Phase 4 on its own.
* **Combo D Test Completed:** We ran a combined test of **Phase 2 (Stationary Features)** + **Phase 3 (Regularized Models)** to see if we could achieve strong performance without relying on the US-centric short volume data. 

## Final Gate 2 Results (Combo D)
* **Status:** `[FAIL]`
* **AUC Improvement:** +0.0032 (Solid win over baseline).
* **Variance/Standard Deviation:** +0.0133 (Failed. Performance became too inconsistent across different tickers).
* **Random Chance Gate:** 1 ticker dropped below 0.50 AUC (Failed).

## What to do Next (The Handoff)
Whenever work resumes on this project, the immediate next steps should be:

1. **Investigate the Failing Ticker:** Find out which single ticker dropped below 0.50 in the Combo D test (`gate2_results.csv` or pipeline logs) and determine why.
2. **Address the Variance Spike:** Look into why the Ridge/Lasso models caused the standard deviation to increase. This usually means the models severely over-penalized some tickers while boosting others. Tuning the regularization hyperparameters (`alpha`) could fix this.
3. **Test Alternative Combos:** 
   * Try **Combo A** (Stationary Features + Short Volume).
   * Evaluate if it's worth running dual pipelines: one dedicated ML pipeline for US equities (with Short Volume) and a separate one for EU equities. 

*References:*
* Detailed execution combos can be found in `open-issues/solutions_todo.md` (or the `solutions_todo.md` artifact).
* EU data investigation notes are in `open-issues/01_eu_short_data.md`.
