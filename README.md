# nimitha-uwsn-system
## Federated Learning with Digital Twin Drift Adaptation

## 1. Problem Description
FL system with Digital Twin deviation detector and Continual Learning adaptation.

## 2. Setup
pip install -r requirements.txt

## 3. How to Run
python src/main.py

## 4. Experiment Mapping
Table 1 -> experiments/exp1_baseline.py
Figure 2 -> experiments/exp2_no_adapt.py
Figure 3 -> experiments/exp3_no_fl.py
Detection -> experiments/exp4_drift.py

## 5. Expected Output
[ROUND 01] Acc=0.820 | Dev=0.2513
[ROUND 09] Acc=0.590 | Dev=0.9100  <- DRIFT DROP
[ROUND 15] Acc=0.810 | Dev=0.1500  <- RECOVERY
