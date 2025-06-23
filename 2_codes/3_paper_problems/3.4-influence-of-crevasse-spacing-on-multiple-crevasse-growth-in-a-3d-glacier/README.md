3.4. Influence of Crevasse Spacing on Multiple-Crevasse Growth in a 3D Glacier

In this section, Im gonna to implement 3D single crack simulations. Below is the problem setup. The domain has size of 750 m x 500 m x125 m, and is subjected a gravity. The right surface is subjected a hydrostatic pressure caused by water. In this study, we select the level of water (hw =0.5). The top surface is free. Other surfaces are applied roller BCs as below. To trigger the damage there are 5 notches as shown in the figure.  Spacing between adjacent notches is L. I do parametric study with S = 50, 70 m to see how the spacing L impact crack propagation



![](attachments/Pasted%20image%2020250623115755.png)

![](attachments/Pasted%20image%2020250623121713.png)

Here are codes, 5mul.py and 7mul are corresponding with S = {50, 70 m}, respectively.