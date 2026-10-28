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
Research paper and optimization code for maximizing efficiency and minimizing passenger bottlenecks at airport security checkpoints.
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class AirportSecuritySimulation:
    def __init__(self, data_path: str):
        """
        Initializes the dynamic agent-based structural engine from dataset parameters.
        Loads conditional distributions to bind simulation probabilities to data trends.
        """
        # Load dataset and drop empty terminal target vectors safely
        self.df = pd.read_csv(data_path).dropna(subset=['Security_Threat_Level'])
        
        # Empirical Parameter Extractor
        self.total_time_steps = len(self.df)
        
        # Calculate mean processing metrics dynamically from data fields
        self.mean_screening_time = self.df['Screening_Time'].mean() if 'Screening_Time' in self.df else 3.15
        self.mean_preparedness = self.df['Passenger_Preparedness'].mean() if 'Passenger_Preparedness' in self.df else 0.72
        self.mean_staff_experience = self.df['Security_Staff_Experience'].mean() if 'Security_Staff_Experience' in self.df else 14.5
        
        # Calculate background base arrival parameters across the execution envelope
        if 'Passenger_Throughput_Level' in self.df:
            self.base_arrival_rate = self.df['Passenger_Throughput_Level'].dropna().mean() + 1.0
        else:
            self.base_arrival_rate = 1.5

    def run_agent_engine(self, total_servers: int) -> dict:
        """
        Executes a localized stochastic agent-based structural run.
        Tracks individual micro-delays and cumulative system performance.
        """
        np.random.seed(42) # Establish baseline seed uniformity across configurations
        
        # Dynamic capacity metrics calculation
        # Higher staff experience and passenger preparedness compress active bottleneck cycles
        efficiency_multiplier = (1.0 + (self.mean_preparedness * 0.15)) * (1.0 + (self.mean_staff_experience * 0.02))
        service_rate_per_server = (1.0 / self.mean_screening_time) * efficiency_multiplier
        total_service_capacity = total_servers * service_rate_per_server
        
        # Analytical metric array allocation
        queue_backlog = 0.0
        processed_passengers = 0
        total_waiting_time = 0.0
        unresolved_threat_backlog = 0.0
        
        # Extract system arrays for optimized step-by-step vector streaming
        ai_risk_scores = self.df['AI_Risk_Score'].fillna(self.df['AI_Risk_Score'].median()).values
        prohibited_items = self.df['Prohibited_Item_Detected'].fillna(0).values
        baggage_results = self.df['Baggage_Screening_Result'].fillna(0).values
        
        # Execution loop matching len(df) iterations
        for t in range(self.total_time_steps):
            # 1. Micro-Agent Arrival injection phase (stochastic modeling)
            step_arrival_intensity = np.random.poisson(self.base_arrival_rate)
            queue_backlog += step_arrival_intensity
            
            # 2. Extract agent risk profile definitions matching specific data indexes
            current_risk_score = ai_risk_scores[t % len(ai_risk_scores)] / 100.0
            
            # 3. Probabilistic logic execution step [Formula: baseline + normalized risk weight]
            threat_probability = 0.05 + 0.30 * current_risk_score
            
            # Determine if current incoming batch introduces structural risk anomalies
            generated_threats = 0
            for _ in range(step_arrival_intensity):
                # Classify if candidate agent maps directly to threat signatures
                is_threat = (np.random.rand() < threat_probability) or (prohibited_items[t % len(prohibited_items)] == 1)
                if is_threat:
                    generated_threats += 1
            
            # 4. Service execution processing block
            actual_processed = min(queue_backlog, total_service_capacity)
            queue_backlog -= actual_processed
            processed_passengers += actual_processed
            
            # Record structural wait anomalies accumulated at current scale
            total_waiting_time += queue_backlog
            
            # 5. Security agent classification and detection resolution pass
            # Base model matching YOLO-style detection performance (75% precision matrix)
            detection_probability = 0.75
            detected_threats = np.round(generated_threats * detection_probability)
            unresolved_threat_backlog += (generated_threats - detected_threats)
            
        # Compile processing vectors into standard metrics maps
        return {
            "Throughput": int(processed_passengers),
            "Average_Queue_Length": float(total_waiting_time / self.total_time_steps),
            "Bottleneck_Risk_Index": float(queue_backlog),
            "Unresolved_Threat_Leakage": int(max(0, unresolved_threat_backlog))
        }

    def structural_solver_objective(self, x) -> float:
        """
        Objective mathematical function minimized by the optimizer framework solver.
        Balances financial/staff operational resource costs against bottleneck waiting times.
        """
        allocated_servers = max(1, int(np.round(x[0])))
        metrics = self.run_agent_engine(total_servers=allocated_servers)
        
        # Loss Penalties Configuration
        bottleneck_penalty = metrics["Average_Queue_Length"] * 50.0
        resource_cost_penalty = allocated_servers * 120.0
        threat_leakage_penalty = metrics["Unresolved_Threat_Leakage"] * 500.0  # High penalty for security breaches
        
        return bottleneck_penalty + resource_cost_penalty + threat_leakage_penalty

    def execute_global_optimization(self) -> dict:
        """
        Runs a constrained bounded solver tracking objective system equilibrium.
        """
        print("[!] Initializing Agent-Based Operational Bottleneck Analysis Pipeline...")
        
        # Define operational boundaries (min: 1 server, max: 20 checkpoint lanes)
        bounds = [(1, 20)]
        initial_guess = [3]  # Start checking configurations from 3 open lanes
        
        # Execute non-linear minimization constraint optimization pass
        solver_result = minimize(
            self.structural_solver_objective, 
            initial_guess, 
            method='Powell', # Derivative-free step optimization for stochastic simulation functions
            bounds=bounds
        )
        
        optimized_lane_count = max(1, int(np.round(solver_result.x[0])))
        
        # Extract performance comparatives using optimized vs unoptimized lane baselines
        baseline_performance = self.run_agent_engine(total_servers=2)
        optimized_performance = self.run_agent_engine(total_servers=optimized_lane_count)
        
        return {
            "optimal_lanes_required": optimized_lane_count,
            "baseline_metrics_2_lanes": baseline_performance,
            "optimized_metrics": optimized_performance,
            "convergence_status": "Success" if solver_result.success else "Partial"
        }

# --- System Deployment Execution Execution block ---
if __name__ == "__main__":
    # Standard fallback simulation using mock internal structures if localized path needs matching
    try:
        # Pass file directly to initialization constructor
        pipeline = AirportSecuritySimulation(data_path="airport_security_dataset.csv")
        results = pipeline.execute_global_optimization()
        
        print("\n========================================================")
        print("          OPTIMIZATION COMPILATION REPORT               ")
        print("========================================================")
        print(f"[*] Optimal Active Screening Lanes Recommended : {results['optimal_lanes_required']} lanes")
        print(f"[*] Core Processing Pipeline Status           : {results['convergence_status']}")
        print("\n--- BASELINE OPERATIONAL PRESSURE (2 LANES DEFAULT) ---")
        print(f" -> Mean System Queue Backlog Accumulation: {results['baseline_metrics_2_lanes']['Average_Queue_Length']:.2f} agents/step")
        print(f" -> Bottleneck Residual Scale At End Time: {results['baseline_metrics_2_lanes']['Bottleneck_Risk_Index']:.2f}")
        print(f" -> Unresolved Risk Anomalies Leakage Rate: {results['baseline_metrics_2_lanes']['Unresolved_Threat_Leakage']} items")
        print("\n--- OPTIMIZED ENVELOPE METRICS (SOLVER CONVERGED) ---")
        print(f" -> Mean System Queue Backlog Accumulation: {results['optimized_metrics']['Average_Queue_Length']:.2f} agents/step")
        print(f" -> Bottleneck Residual Scale At End Time: {results['optimized_metrics']['Bottleneck_Risk_Index']:.2f}")
        print(f" -> Unresolved Risk Anomalies Leakage Rate: {results['optimized_metrics']['Unresolved_Threat_Leakage']} items")
        print("========================================================")
        
    except FileNotFoundError:
        print("[!] Target file verification check failed. Please save the file raw data structure as 'airport_security_dataset.csv'.")
