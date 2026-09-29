# Synthetic heat-treatment process card

This process is a two-dimensional cell-centred finite-volume model of a
rectangular workpiece. The unmodelled front and back faces are insulated;
`thickness` converts in-plane edge lengths into physical boundary areas. All
coefficients are synthetic design assumptions. They are constant within one
trajectory and may differ between trajectories.

The field obeys transient heat conduction. Every in-plane boundary face is
assigned to exactly one heating zone. Its convective and radiative heat flux
passes through a half-cell conduction resistance. The identical discrete flux
enters the solid and leaves the zone, giving a checkable total-energy balance.
Heating-zone capacitance, electric power efficiency, ambient loss, and
symmetric inter-zone coupling are explicit. Radiation uses kelvin.

The solver uses implicit Euler and fixed-point iteration of the nonlinear
radiative conductance. It raises on nonconvergence or invalid temperatures;
unconverged states are never returned as training labels. The user must set
grid and internal integration step separately from the control period.

This model is not calibrated to a furnace. It has no phase transitions,
temperature-dependent properties, 3D edge effects, sensor calibration, or
hardware interlocks. Its energy-balance tests check implementation consistency,
not industrial predictive accuracy.
