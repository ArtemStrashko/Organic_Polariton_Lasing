The data and scripts are organised as follows:

* data_and_scripts_for_producing_figures/

This contains python scripts to produce the figures in the paper, along with the data (as a python pickle) required for these figures.  For the structure of the data in each file, see the scripts for plotting figures.  (Filenames indicate which figures they produce).

* scripts_for_producing_data/

This contains python scripts that produce the data.  Unnumbered scripts contain definitions of functions etc, while numbered scripts correspond to producing the data for specific figures in the paper. 

----------------------------------------------------------------------
Further details on the scripts are as follows:

Unnumbered scripts

* GGM.py
Defines Gell-Mann matrices.

* Coefficients_rwa_non_rwa.py
Calculate Hamiltonian and Lindblad parameters when rewriting equations in the basis of Gell-Mann matrices. Calculates other coefficients appearing in equations.

* MF_equations__with_A_squared.py
Defines stability matrix with and without A^2 term and also a function to find whether an eigenmode is unstable or not.

* stability_along_boundary.py
Calculates density matrix components at the normal-lasing boundary and the positions of corresponding pumping Gam_up versus bare photon frequency om_c.

* stability_for_gam_up_slices.py
Calculates density matrix components vs bare phot frequency for a given value of pumping Gam_up (for a given slice) for plotting spectrum and density matrix components (which molecular transitions are involved).

* WCT_equations.py
Calculates emission/absorption spectrum, defines weak coupling equations, calculates steady-state solution of these equations, a function to find an element index.



Numbered scripts


* 1_Abs_emis_spectra__fig_1b.py
Calculates absorption/emission spectra


* 2_Phase_diagr__Pump__phot_freq__fig2_S1_S2_S3_S4top.py
Calculates phase diagram versus pumping strength and bare photon frequency

- To get phase diagrams at different matter-light coupling change the parameter g_light_mat and keep in mind that in the mean-field approach only g_light_mat * sqrt(N_m) controles physics. So, here it's enough to keep N_m=1 and change g_light_mat only. All other parameters have their obvious meaning and are described by comments.

- To plot phase diagrams taking into account (ignoring) A^2 term, comment (decomment) a corresponding line appearing under "# build stability matrix ..." leaving either "M = M_stability_with_A_squared..." or "M = M_stability" respectively.


* 3_Phase_diagrams__Pump__coupl_for_fig4abc_S7top.py
Calculates phase diagram versus pumping and matter-light coupling

- To get phase diagrams at different bare photon frequency change a parameter om_c

- Regarding keeping/ignoring A^2 term - the same as described above


* 4_Spectum_and_DM_components_for_fig3_S3_S4.py
Calculates spectrum and molecular transitions involved in lasing along pumping Gam_up slice versus bare photon frequency om_c

- Choose matter-light coupling changing g_light_mat and keeping N_m=1 (as in the second script "2_Phase_diagr__Pump...")

- Choose slices of pumping you want to explore (array Gam_up_array)

- Run the code

- Go to the line "# choose the number of Gamma_up slice, for which you want to plot spectrum". Choose the number of the pumping Gam_up slice, for which you want to plot a spectrum, and then run the piece of code under the line "# choose the number of Gamma_up slice, for which you want to plot spectrum" and "# plotting/saving results for the SPECTRUM". Change the number of Gam_up slice and repeat the steps above. 

- Go to the section "## Density Matrix components of an unstable mode". Choose the number of pumping slice under "# choose the number of Gamma_up slice" and run this last part of the code to get molecular transitions involved in lasing.


* 5_WCT_analyt_bound_line_fig4.py
Calculates analytical weak coupling theory normal-lasing state boundary

- Choose bare photon frequency om_c and run the code


* 6_DM_components_along_boundary_for_fig1d.py
Calculates the composition of molecular transitions involved in lasing along normal-lasing boundary

- Follow similar steps as in the scripts above.


* 7_crit_pump_for_lasing__opt_freq__bisection_for_fig_S5.py
Calculates the optimal (for lowest lasing threshold) photon frequency vs matter-light coupling.

This script is finely tuned to follow the position of all the local minima specifically for the parameters used in the paper.  To study other parameters requires fine tuning to make sure one captures all minima.






