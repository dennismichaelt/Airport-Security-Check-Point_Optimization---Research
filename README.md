# Airport-Security-Check-Point_Optimization---Research

Built by: Anthony Christopher P. and Dennis Michael T.

We built an Agent-Based Model (ABM) in Python to simulate how passenger traffic bottlenecks form at airport security checkpoints. The goal was to see how technical factors (like AI risk scoring models) and human variables (like staff experience and how prepared passengers are) interact when a checkpoint gets slammed during peak hours.

# About the Dataset and Data Engineering
Since real aviation security data is highly restricted and confidential, we engineered our own synthetic operational dataset (roughly 16.7k rows) to run our simulations. 

We did not just randomize it; we mapped out specific equations so the simulation would behave realistically. For instance, a passenger's baseline threat probability is calculated using their risk vectors:
threat_probability = 0.05 + 0.3 * normalized_risk

What the dataset tracks:
- Human elements: Passenger_Preparedness (do they have their laptops out?), Security_Staff_Experience (in years).
- System elements: AI_Risk_Score (normalized 0-1 metrics), Baggage_Screening_Result, and overall Passenger_Throughput_Level.

## The Tech Stack
- Language: Python 3
- Libraries used: pandas and numpy for handling arrays and simulation math. matplotlib and seaborn for plotting the final distribution graphs. scikit-learn for running baseline threat classification tests.

## What We Learned (Key Takeaways)
1. The biggest levers: Not surprisingly, Prohibited_Item_Detected and Baggage_Screening_Result had the strongest direct correlation with threat classification.
2. Staff experience matters: In our simulation loops, higher security staff experience levels directly curbed false positive classifications, smoothing out traffic flow.
3. System failure thresholds: When Passenger_Throughput_Level crosses peak thresholds, unresolved threats start accumulating exponentially if the automated detection flags lag behind, causing the global threat line to spike upward.

## Repository Contents
- *.csv: The engineered dataset we ran our loops against.
- *.pdf / *.pptx: Our presentation slides, methodology notes, and final graphical analysis charts.
-
