# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 23:58:07 2025

@author: Tiger
"""

# Libraries
import pandas as pd
import networkx as nx
from dowhy import CausalModel
from sklearn.impute import  SimpleImputer
import category_encoders as ce
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self, df):
        self.df = df

    def handle_missing(self, choice):
        if choice == '1':
            self.df = self.df.dropna()
        elif choice == '2':
            imputer = SimpleImputer(strategy='most_frequent')
            self.df = pd.DataFrame(imputer.fit_transform(self.df))
        return self.df

    def select_variables(self, choice):
        if choice == '1':
            self.df = pd.DataFrame({
                    'education': self.df.iloc[:,3],
                    'occupation': self.df.iloc[:, 6],
                    'income': self.df.iloc[:,14]
                })
        elif choice == '2':
            self.df = pd.DataFrame({
                    'education': self.df.iloc[:,3],
                    'age' : self.df.iloc[:,0],
                    'occupation': self.df.iloc[:, 6],
                    'income': self.df.iloc[:,14]
                })
        return self.df

    def select_treatment_type(self, choice):
        if choice == "1":
            self.df['occupation'] = self.df['occupation'].apply(lambda x: "Exec-managerial" if x.strip() == "Exec-managerial" 
                                                      else "Default")
        elif choice == "2":
            self.df['occupation'] = self.df['occupation'].apply(lambda x: x.strip())
            self.df = self.df[self.df['occupation'].isin(["Exec-managerial", "Prof-specialty"])]
        return self.df

    def encode_features(self, choice):
        if choice == '1':
            self.df = pd.get_dummies(self.df, columns=['income', 'occupation'], drop_first=True)
        elif choice == '2':
            encoder = ce.OneHotEncoder(use_cat_names=True)
            self.df = encoder.fit_transform(self.df)

        # Optional: create single treatment/outcome column
        if 'occupation_Exec-managerial' in self.df.columns:
            self.df['occupation'] = self.df['occupation_Exec-managerial']
        if 'income_ >50K' in self.df.columns:
            self.df['income'] = self.df['income_ >50K']

        return self.df
    
    
    
class CausalGraphBuilder:
    def __init__(self):
        self.graph_string = None
        self.graph_nx = None

    def define_dag(self, choice, df):
        """
        Builds DAG either manually (choice 1) or using NetworkX (choice 2).
        Returns both:
          - graph_nx: NetworkX DiGraph (for CausalModel)
          - graph_string: DOT string (for Streamlit visualization)
        """
        # Common edges
        edges = [
            ('education', 'occupation'),
            ('education', 'income'),
            ('occupation', 'income')
        ]
    
        # Include age if present
        if any(col.startswith('age') for col in df.columns):
            edges += [('age', 'occupation'), ('age', 'income')]
    
        # Choice 1: Manual DAG string
        if choice == '1':
            nodes = ['education', 'occupation', 'income']
            if any(col.startswith('age') for col in df.columns):
                nodes.insert(1, 'age')  # insert age after education
    
            # Build DOT string manually
            dot_str = "graph [\n\tdirected 1\n"
            for node in nodes:
                dot_str += f'\tnode [id "{node}" label "{node}"]\n'
            for src, tgt in edges:
                dot_str += f'\tedge [source "{src}" target "{tgt}"]\n'
            dot_str += "]"
            self.graph_string = dot_str
    
            # Also create NetworkX graph for causal model
            self.graph_nx = nx.DiGraph()
            self.graph_nx.add_edges_from(edges)
    
        # Choice 2: Auto DAG with NetworkX
        elif choice == '2':
            self.graph_nx = nx.DiGraph()
            self.graph_nx.add_edges_from(edges)
            # Convert NetworkX to DOT string
            self.graph_string = nx.nx_pydot.to_pydot(self.graph_nx).to_string()
    
        # Return both for downstream tasks
        return self.graph_nx, self.graph_string


    def draw_graph(self):
        if self.graph_nx is None:
            print("No Graph present. Please run the define_dag first.")
            return
        plt.figure(figsize=(6,4))
        pos = nx.spring_layout(self.graph_nx, seed=42)
        nx.draw(self.graph_nx, pos, with_labels=True, node_size=2000,
               node_color='lightblue', font_size=10, font_weight="bold", arrows=True)
        plt.title("Causal DAG")
        plt.show()
        
        
        
class CausalModelManager:
    def __init__(self, df, treatment, outcome, graph):
        self.model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=graph
        )
        self.estimand = None
        self.estimate = None

    def identify_effect(self, choice):
        if choice == '1':
            self.estimand = self.model.identify_effect()
        elif choice == '2':
            self.estimand = self.model.identify_effect(proceed_when_unidentifiable=True, method_name='frontdoor')
        
        print("\nIdentifies Estimand:")
        print(self.estimand)
        return self.estimand

    def estimate_effect(self, choice):
        if self.estimand is None:
            raise ValueError("Estimand is not identifies yet. Run identify_effect()")
        
        if choice == '1':
            self.estimate = self.model.estimate_effect(
                identified_estimand=self.estimand,
                method_name = "backdoor.linear_regression"
            )
        elif choice == "2":
            self.estimate = self.model.estimate_effect(
                identified_estimand=self.estimand,
                method_name="backdoor.propensity_score_matching"
            )

        print(f"\nEstimated Causal Effect: {self.estimate.value}")
        return self.estimate

    def refute_effect(self):
        if self.estimand is None or self.estimate is None:
            raise ValueError("You must identify and estimate before refuting.")

        print("\nRunning refutation tests...")
        refute1 = self.model.refute_estimate(self.estimand, self.estimate, method_name="random_common_cause")
        refute2 = self.model.refute_estimate(self.estimand, self.estimate, method_name="placebo_treatment_refuter")

        # print("\nRefutation 1 (Random Common Cause):")
        # print(refute1)
        # print("\nRefutation 2 (Placebo Treatment):")
        # print(refute2)
        return (refute1, refute2)