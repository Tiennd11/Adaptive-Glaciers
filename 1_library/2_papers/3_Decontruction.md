- Motivation ( big challenges/ Ice fracturing) 
	- There is a need for computational models capable of predicting meltwater-assisted crevasse. 
	- Mass loss from glaciers and ice sheets is the largest contributor to sea- level rise and iceberg calving due to hydrofracture is one of the most prominent yet less understood glacial mass loss process growth in glacial ice.
- Purpose (How we solve the challenges ( phase field ....) )
	- To overcome the limitations of empirical and analytical approaches, we here propose a new phase field-based computational framework to simulate crevasse growth in both grounded ice sheets and floating ice shelves

- Method (Steps to solve the problem)
	- Numerical method FEM + Phase field ( Coupled framework)
	- First, the influence of the choice of material rheology and relevant parameters are investigated by simulating the propagation of a single crevasse in grounded glaciers
	- Second, crevasse interaction is assessed by predicting the growth of a field of densely spaced crevasses in a grounded glacier
	- The third case study addresses the interaction between surface and basal crevasses in a floating ice shelf, appropriately simulated using Robin boundary conditions
	- Nucleation and growth of crevasses in a realistic geometry, that of the Helheim glacier, is predicted in the fourth case study, combining a sequential creep-damage analysis. 
	- Finally, the last case study provides the first simulation of interacting crevasses in 3D ice sheets
- Originalities ( New in the work/ How the work is different from)
	- The work describes a constitutive description incorporating the non-linear viscous rheology of ice
	- The work describes a phase field formulation capable of capturing cracking phenomena of arbitrary complexity, such as 3D crevasse interaction
	- a poro-damage representation to account for the role of meltwater pressure on crevasse growth
- Value (Accuracy/ computational cost)
	- Model predictions provide a good agreement with LEFM and Nye’s zero stress model when particularised to the idealised conditions where these analytical approaches are relevant.
	- The model adequately predicts the propagation of crevasses in regions where the net longitudinal stress is tensile, without the need for ad hoc fracture driving force decompositions and exhibiting very little sensitivity to the choice of phase field length scale 𝓁

- Finding (What are the findings of the work)
	- Increasing amounts of meltwater, as a result of climate change, can significantly enhance crevasse propagation, with iceberg calving being predicted for meltwater depth ratios of 50% or larger.
	- Predicted crevasse depths are greater when considering the incompressible stress state intrinsic to a non-linear viscous rheology. Thus, first-order estimates obtained from analytical LEFM approaches should consider a Poisson’s ratio of 𝜈 = 0.5 to avoid underpredicting the impact of meltwater on ice-sheet stability
	- The model captures how the presence of neighbouring surface crevasses provides a shielding effect on the stress concentration and reduces the predicted crevasse depth.
	- The model accurately predicts the growth of surface crevasses within floating ice shelves near the shelf front for large meltwater depth ratios. Also, if a surface crevasse is in close proximity to a basal crevasse then a reduction in basal crevasse penetration depth is observed
	- Crevasses are predicted to nucleate in areas with high surface gradients, highlighting the need for an adequate characterisation of the glacier’s geometry
	- The large-scale 3D analyses conducted demonstrate the capabilities of the model of opening new horizons in the modelling of crevasse growth phenomena under the computationally-demanding conditions relevant to iceberg calving

- Limitations
	- Don't know the fracture location/ prerefine mesh can be a problem. Adaptive can solve the problem.
	
	- 