# European Short Volume Data - Coverage Gap

## The Issue
During Phase 4 (Alternative Data), we successfully integrated FINRA RegSHO daily short volume data for the US-listed equities. This resulted in 77 covered tickers. 

However, approximately 46 European tickers (e.g. `SAP.DE`, `BAYN.DE`, `VOW3.DE`) dropped out of the Phase 4 test. 

## Investigation Summary
We investigated alternative sources to obtain daily short volume for European equities to close this coverage gap:
1. **ESMA Short Positions Register**: Only publishes *net* short positions that exceed the 0.5% reporting threshold, not daily short volume.
2. **BaFin (Germany)**: Same as ESMA (net positions > 0.5%).
3. **Deutsche Börse / Xetra**: No free public API for short volume.
4. **shortsell.nl**: We successfully scraped this site, but it only contains the *latest* snapshot (or a few recent data points) of the net short position. The ML pipeline requires a continuous daily time-series (e.g., 63 days) to compute rolling means and z-scores (like `sv_short_ratio_z21`).

## Conclusion
Due to EU Short Selling Regulation (SSR) differences, daily short *volume* is not published the way it is in the US via FINRA. The model handles this gracefully by dropping the `sv_` features for European tickers.

## Next Steps
This issue is currently shelved. We are proceeding with "Combo D" (Stationary Features + Regularized Models) which impacts all 123 tickers positively without relying on the alternative data, while allowing the pipeline to safely ignore missing `sv_` columns for EU tickers when Short Volume is enabled later.
