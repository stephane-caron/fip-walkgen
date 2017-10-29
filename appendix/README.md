## Appendices

Code used in the appendices of the paper.

### Appendix A: Proof of Proposition 1

Run ``python cone_calculations.py | pdflatex`` to generate a PDF document with
all intermediate wrench cone calculations.

### Appendix B: Support Volumes for Virtual Repulsors and Attractors

The script ``multi_contact_cones.py`` provides functions to compute and draw
the cones of COM positions described in this appendix:

- ``compute_cop_pendular_cone`` and ``draw_cop_pendular_cone``: when the center
  of pressure (COP) is fixed.
- ``compute_vrp_com_cone`` and ``draw_vrp_com_cone``: when the virtual
  repellent point (VRP) is fixed.
- ``compute_attraction_polyhedron`` and ``draw_attraction_polyhedron``: when
  the virtual attractor point (VAP) is fixed.
