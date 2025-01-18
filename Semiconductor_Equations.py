import numpy as np

# Constants
vac_permittivity = 8.85e-14  # F/cm
freespace_permeability = np.pi * 4e-7  # H/m
h = 6.63e-34  # J*s
h_bar = h / (2 * np.pi)
q = 1.6e-19  # C
k_B = 1.38e-23  # J/K
c = 3e10  # cm/s
m_0 = 9.11e-31  # kg
kT = 0.0259  # eV at room temperature (approximately 300K)

# Parabolic Bands Assumption Functions
def Nc(T, m_c):
    """Calculate the effective density of states in the conduction band.
   
    Parameters:
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
       
    Returns:
        float: Density of states in m⁻³

    """
    return 2 * ((2 * np.pi * m_c * m_0 * k_B * T) / h**2)**1.5

def Nv(T, m_v):
    """Calculate the effective density of states in the valence band.
   
    Parameters:
        T (float): Temperature in Kelvin
        m_v (float): Effective mass of valence electrons (dimensionless)
       
    Returns:
        float: Density of states in m⁻³

    """
    return 2 * ((2 * np.pi * m_v * m_0 * k_B * T) / h**2)**1.5

def law_mass_action(E_g, T, m_c, m_v):
    """Calculate the product of electron and hole concentrations (law of mass action).
   
    Parameters:
        E_g (float): Energy gap in eV
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
       
    Returns:
        float: Product of concentrations in m⁻⁶

    """
    return Nc(T, m_c) * Nv(T, m_v) * np.exp((-E_g * q) / (k_B * T))

# Intrinsic Semiconductors
def intrinsic_concentration_n_i(E_g, T, m_c, m_v):
    """Calculate the intrinsic carrier concentration in an intrinsic semiconductor.
   
    Parameters:
        E_g (float): Energy gap in eV
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
       
    Returns:
        float: Intrinsic carrier concentration in m⁻³

    """
    return (Nc(T, m_c) * Nv(T, m_v))**0.5 * np.exp((-E_g * q) / (2 * k_B * T))

def E_fermi_intrinsic(E_g, m_c, m_v):
    """Calculate the Fermi level for intrinsic semiconductors with respect to valence energy.
   
    Parameters:
        E_g (float): Energy gap in eV
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
       
    Returns:
        float: Fermi energy level in eV

    """
    return 0.5 * E_g + 0.75 * kT * np.log(m_v / m_c)

# Extrinsic Semiconductors
def E_fermi_extrinsic_n(E_g, m_c, m_v, N_D, T):
    """Calculate the Fermi level for n-type extrinsic semiconductors.
   
    Parameters:
        E_g (float): Energy gap in eV
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
        N_D (float): Donor density in cm⁻³
        T (float): Temperature in Kelvin
       
    Returns:
        float: Fermi energy level in eV

    """
    n_i = intrinsic_concentration_n_i(E_g, T, m_c, m_v)
    E_f_intrinsic = E_fermi_intrinsic(E_g, m_c, m_v)
    return E_f_intrinsic + kT * np.log(N_D / n_i)

def E_conduction_E_fermi_extrinsic_n_delta(T, m_c, N_D):
    """Calculate the energy difference between conduction band edge and Fermi level for n-type semiconductor.
   
    Parameters:
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
        N_D (float): Donor density in cm⁻³
       
    Returns:
        float: Energy difference in eV

    """
    return (k_B * T / q) * np.log(Nc(T, m_c) / N_D)

def E_fermi_extrinsic_p(E_g, m_c, m_v, N_A, n_i):
    """Calculate the Fermi level for p-type extrinsic semiconductors.
   
    Parameters:
        E_g (float): Energy gap in eV
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
        N_A (float): Acceptor density in cm⁻³
        n_i (float): Intrinsic carrier concentration in cm⁻³
       
    Returns:
        float: Fermi energy level in eV

    """
    return E_fermi_intrinsic(E_g, m_c, m_v) - kT * np.log(N_A / n_i)

def E_fermi_extrinsic_p_E_valence_delta(T, m_v, N_A):
    """Calculate the energy difference between valence band edge and Fermi level for p-type semiconductor.
   
    Parameters:
        T (float): Temperature in Kelvin
        m_v (float): Effective mass of valence electrons (dimensionless)
        N_A (float): Acceptor density in cm⁻³
       
    Returns:
        float: Energy difference in eV

    """
    return (k_B * T / q) * np.log(Nv(T, m_v) / N_A)

def E_donor(m_c, permittivity):
    """Calculate the ionization energy of donors in a semiconductor.
   
    Parameters:
        m_c (float): Effective mass of conduction electrons (dimensionless)
        permittivity (float): Permittivity of the material (dimensionless)
       
    Returns:
        float: Donor ionization energy in eV

    """
    return 13.6 * (m_c * m_0 / m_0) * (1 / permittivity)**2

# Charge Neutrality Functions
def delta_n(set_num, n_C, p_v, N_A, N_D):
    """
    Calculate charge neutrality (fully ionized impurities).
   
    Parameters:
        set_num (int): 1 for delta_n as n_C - p_v, 2 for delta_n as N_D - N_A
        n_C (float): Conduction band electron density in cm⁻³
        p_v (float): Valence band hole density in cm⁻³
        N_A (float): Acceptor density in cm⁻³
        N_D (float): Donor density in cm⁻³
       
    Returns:
        float: Resulting delta_n in cm⁻³
    """
    if set_num == 1:
        return n_C - p_v
    elif set_num == 2:
        return N_D - N_A
    else:
        raise ValueError("Invalid set number. Use 1 or 2.")

def ionized_intrinsic_concentration_n_i(n_C, p_v):
    """Calculate intrinsic carrier concentration with ionized impurities.
   
    Parameters:
        n_C (float): Conduction band electron density in cm⁻³
        p_v (float): Valence band hole density in cm⁻³
       
    Returns:
        float: Intrinsic carrier concentration in cm⁻³

    """
    return (n_C * p_v)**0.5

def electron_density_conduction_band(set_num, n_C, p_v, N_A, N_D):
    """Calculate electron density in the conduction band.

    Parameters:
        set_num (int): Choice of calculation
        n_C (float): Conduction band electron density in cm⁻³
        p_v (float): Valence band hole density in cm⁻³
        N_A (float): Acceptor density in cm⁻³
        N_D (float): Donor density in cm⁻³

    Returns:
        float: Electron density in cm⁻³

    """
    intrinsic_n_i = ionized_intrinsic_concentration_n_i(n_C, p_v)
    delta_n_val = delta_n(set_num, n_C, p_v, N_A, N_D)
    return 0.5 * ((delta_n_val**2 + 4 * intrinsic_n_i**2)**0.5 + delta_n_val)

def electron_density_valence_band(set_num, n_C, p_v, N_A, N_D):
    """
    Calculate hole density in the valence band.

    Parameters:
        set_num (int): Choice of calculation
        n_C (float): Conduction band electron density in cm⁻³
        p_v (float): Valence band hole density in cm⁻³
        N_A (float): Acceptor density in cm⁻³
        N_D (float): Donor density in cm⁻³

    Returns:
        float: Hole density in cm⁻³
    """
    intrinsic_n_i = ionized_intrinsic_concentration_n_i(n_C, p_v)
    delta_n_val = delta_n(set_num, n_C, p_v, N_A, N_D)
    return 0.5 * ((delta_n_val**2 + 4 * intrinsic_n_i**2)**0.5 - delta_n_val)

def low_impurity_concentration_n(n_C, p_v, N_A, N_D):
    """
    Calculate electron density at low impurity concentration.
   
    Parameters:
        n_C (float): Conduction band electron density in cm⁻³
        p_v (float): Valence band hole density in cm⁻³
        N_A (float): Acceptor density in cm⁻³
        N_D (float): Donor density in cm⁻³
       
    Returns:
        float: Electron density in cm⁻³
    """
    intrinsic_n_i = ionized_intrinsic_concentration_n_i(n_C, p_v)
    return intrinsic_n_i + 0.5 * (N_D - N_A)

def low_impurity_concentration_p(n_C, p_v, N_A, N_D):
    """
    Calculate hole density at low impurity concentration.
   
    Parameters:
        n_C (float): Conduction band electron density in cm⁻³
        p_v (float): Valence band hole density in cm⁻³
        N_A (float): Acceptor density in cm⁻³
        N_D (float): Donor density in cm⁻³
       
    Returns:
        float: Hole density in cm⁻³
    """
    intrinsic_n_i = ionized_intrinsic_concentration_n_i(n_C, p_v)
    return intrinsic_n_i - 0.5 * (N_D - N_A)

def high_impurity_concentration_n(N_A, N_D):
    """
    Calculate electron density at high impurity concentration (N_D >> N_A).
   
    Parameters:
        N_A (float): Acceptor density in cm⁻³
        N_D (float): Donor density in cm⁻³
       
    Returns:
        float: Electron density in cm⁻³
    """
    return N_D - N_A

def high_impurity_concentration_p(n_C, p_v, N_A, N_D):
    """
    Calculate hole density at high impurity concentration.
   
    Parameters:
        n_C (float): Conduction band electron density in cm⁻³
        p_v (float): Valence band hole density in cm⁻³
        N_A (float): Acceptor density in cm⁻³
        N_D (float): Donor density in cm⁻³
       
    Returns:
        float: Hole density in cm⁻³
    """
    intrinsic_n_i = ionized_intrinsic_concentration_n_i(n_C, p_v)
    return intrinsic_n_i**2 / (N_D - N_A)

# Recombination Rate and Carrier Lifetime Functions
def recombination_rate_R(R_0, n_po, p_po, delta_n, delta_p, E_g, T, m_c, m_v):
    """
    Calculate the recombination rate in a semiconductor.
   
    Parameters:
        R_0 (float): Recombination coefficient (dimensionless)
        n_po (float): Electron concentration at equilibrium (cm⁻³)
        p_po (float): Hole concentration at equilibrium (cm⁻³)
        delta_n (float): Change in electron concentration (cm⁻³)
        delta_p (float): Change in hole concentration (cm⁻³)
        E_g (float): Energy gap in eV
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
       
    Returns:
        float: Recombination rate in cm⁻³/s
    """
    n_i = intrinsic_concentration_n_i(E_g, T, m_c, m_v)
    return (R_0 * (n_po + delta_n) * (p_po + delta_p)) / n_i**2

def minority_carrier_lifetime_tau_n(set_num, R_0, n_po, p_po, B, E_g, T, m_c, m_v):
    """
    Calculate the minority carrier lifetime for electrons.

    Parameters:
        set_num (int): Choice of calculation (1, 2, or 3 for specific cases)
        R_0 (float): Recombination rate (dimensionless)
        n_po (float): Electron concentration at equilibrium (cm⁻³)
        p_po (float): Hole concentration at equilibrium (cm⁻³)
        B (float): Radiative recombination coefficient (dimensionless)
        E_g (float): Energy gap in eV
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
   
    Returns:
        float: Minority carrier lifetime in seconds
    """
    n_i = intrinsic_concentration_n_i(E_g, T, m_c, m_v)
    if set_num == 1:
        return n_po / (B * n_i**2)
    elif set_num == 2:
        return 1 / (B * p_po)
    elif set_num == 3:
        return n_po / R_0
    else:
        raise ValueError("Invalid set number. Use 1, 2, or 3.")

def minority_carrier_lifetime_tau_p(set_num, R_0, n_po, p_po, B, E_g, T, m_c, m_v):
    """
    Calculate the minority carrier lifetime for holes.

    Parameters:
        set_num (int): Choice of calculation (1, 2, or 3 for specific cases)
        R_0 (float): Recombination rate (dimensionless)
        n_po (float): Electron concentration at equilibrium (cm⁻³)
        p_po (float): Hole concentration at equilibrium (cm⁻³)
        B (float): Radiative recombination coefficient (dimensionless)
        E_g (float): Energy gap in eV
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
   
    Returns:
        float: Minority carrier lifetime in seconds
    """
    n_i = intrinsic_concentration_n_i(E_g, T, m_c, m_v)
    if set_num == 1:
        return p_po / (B * n_i**2)
    elif set_num == 2:
        return 1 / (B * n_po)
    elif set_num == 3:
        return p_po / R_0
    else:
        raise ValueError("Invalid set number. Use 1, 2, or 3.")

def delta_recombination_rate_with_lifetime(set_num, delta_n, tau_n, delta_p, tau_p):
    """
    Calculate the recombination rate for charge carriers with specified lifetimes.
   
    Parameters:
        set_num (int): Case selection (1 for electron lifetime, 2 for hole lifetime)
        delta_n (float): Change in electron density (cm⁻³)
        tau_n (float): Electron lifetime (s)
        delta_p (float): Change in hole density (cm⁻³)
        tau_p (float): Hole lifetime (s)
       
    Returns:
        float: Recombination rate in s⁻¹
    """
    if set_num == 1:
        return delta_n / tau_n
    elif set_num == 2:
        return delta_p / tau_p
    else:
        raise ValueError("Invalid set number. Use 1 or 2.")

def radiative_efficiency_eta(radiative_lifetime_tau_R, nonradiative_lifetime_tau_NR):
    """
    Calculate radiative efficiency.
   
    Parameters:
        radiative_lifetime_tau_R (float): Radiative recombination lifetime (s)
        nonradiative_lifetime_tau_NR (float): Nonradiative recombination lifetime (s)
       
    Returns:
        float: Radiative efficiency (dimensionless)
    """
    return 1 / (1 + (radiative_lifetime_tau_R / nonradiative_lifetime_tau_NR))

# Junction Properties
def junction_built_in_voltage_V_bi(set_num, n_no, p_po, ni, E_g, T, m_c, m_v):
    """
    Calculate the built-in voltage for a p-n junction.
   
    Parameters:
        set_num (int): Choice of calculation (1 or 2 for specific cases)
        n_no (float): Electron concentration on n-side (cm⁻³)
        p_po (float): Hole concentration on p-side (cm⁻³)
        ni (float): Intrinsic carrier concentration (cm⁻³)
        E_g (float): Energy gap in eV
        T (float): Temperature in Kelvin
        m_c (float): Effective mass of conduction electrons (dimensionless)
        m_v (float): Effective mass of valence electrons (dimensionless)
       
    Returns:
        float: Built-in voltage in volts
    """
    if set_num == 1:
        ni_value = intrinsic_concentration_n_i(E_g, T, m_c, m_v)
        return (k_B * T / q) * np.log((n_no * p_po) / ni_value**2)
    elif set_num == 2:
        return kT * np.log((n_no * p_po) / ni**2)
    else:
        raise ValueError("Invalid set number. Use 1 or 2.")

def depletion_width_x_n(relative_permittivity, V_bi, donor_conc_Nd):
    """
    Calculate the depletion width on the n-side of a p-n junction.
   
    Parameters:
        relative_permittivity (float): Relative permittivity of the material
        V_bi (float): Built-in voltage in volts
        donor_conc_Nd (float): Donor concentration in cm⁻³
       
    Returns:
        float: Depletion width in cm
    """
    return np.sqrt((2 * relative_permittivity * vac_permittivity * V_bi) / (q * donor_conc_Nd))

def depletion_width_x_p(relative_permittivity, V_bi, acceptor_conc_Na):
    """
    Calculate the depletion width on the p-side of a p-n junction.
   
    Parameters:
        relative_permittivity (float): Relative permittivity of the material
        V_bi (float): Built-in voltage in volts
        acceptor_conc_Na (float): Acceptor concentration in cm⁻³
       
    Returns:
        float: Depletion width in cm
    """
    return np.sqrt((2 * relative_permittivity * vac_permittivity * V_bi) / (q * acceptor_conc_Na))

# Reverse Saturation Current
def reverse_saturation_current_V_sat_n(A, ni, D_p, L_p, N_d):
    """
    Calculate the reverse saturation current for n-type semiconductor.
   
    Parameters:
        A (float): Area in cm²
        ni (float): Intrinsic carrier concentration (cm⁻³)
        D_p (float): Diffusion coefficient of holes (cm²/s)
        L_p (float): Diffusion length of holes (cm)
        N_d (float): Donor concentration in cm⁻³
       
    Returns:
        float: Reverse saturation current in amperes (A)
    """
    return A * q * ni**2 * (D_p / (L_p * N_d))

def reverse_saturation_current_V_sat_p(A, ni, D_n, L_n, N_a):
    """
    Calculate the reverse saturation current for p-type semiconductor.
   
    Parameters:
        A (float): Area in cm²
        ni (float): Intrinsic carrier concentration (cm⁻³)
        D_n (float): Diffusion coefficient of electrons (cm²/s)
        L_n (float): Diffusion length of electrons (cm)
        N_a (float): Acceptor concentration in cm⁻³
       
    Returns:
        float: Reverse saturation current in amperes (A)
    """
    return A * q * ni**2 * (D_n / (L_n * N_a))

def pn_voltage(I_sat, I_diode_applied):
    """
    Calculate the voltage across a p-n junction diode.
   
    Parameters:
        I_sat (float): Reverse saturation current (A)
        I_diode_applied (float): Applied diode current (A)
       
    Returns:
        float: Voltage in volts (V)
    """
    return kT * np.log((I_diode_applied / I_sat) + 1)

# Optical Properties
def fresnel_loss_R_normal(index_material_n_1, index_material_n_2):
    """
    Calculate Fresnel loss for light incident normally on a material interface.
   
    Parameters:
        index_material_n_1 (float): Refractive index of initial material
        index_material_n_2 (float): Refractive index of secondary material
       
    Returns:
        float: Fresnel reflection loss (dimensionless)
    """
    return ((index_material_n_2 - index_material_n_1) / (index_material_n_2 + index_material_n_1))**2

def enhancement(escaping_light_case_1, escaping_light_case_2):
    """
    Calculate enhancement factor for light escaping in different cases.
   
    Parameters:
        escaping_light_case_1 (float): Escaping light intensity in case 1
        escaping_light_case_2 (float): Escaping light intensity in case 2
       
    Returns:
        float: Enhancement factor (dimensionless)
    """
    return escaping_light_case_1 / escaping_light_case_2

def critical_angle_theta_c(index_material_n_1, index_material_n_2):
    """
    Calculate the critical angle for total internal reflection.
   
    Parameters:
        index_material_n_1 (float): Refractive index of initial material
        index_material_n_2 (float): Refractive index of secondary material
       
    Returns:
        float: Critical angle in degrees
    """
    return np.degrees(np.arcsin(index_material_n_2 / index_material_n_1))

def TIR_loss(crit_angle):
    """
    Calculate total internal reflection (TIR) loss.
   
    Parameters:
        crit_angle (float): Critical angle in degrees
       
    Returns:
        float: TIR loss (dimensionless)
    """
    return 0.5 * (1 - np.cos(np.radians(crit_angle)))

# Efficiency
def power_efficiency(power, current, voltage):
    """
    Calculate power efficiency.
   
    Parameters:
        power (float): Power in watts
        current (float): Current in amperes
        voltage (float): Voltage in volts
       
    Returns:
        float: Efficiency (dimensionless)
    """
    return power / (current * voltage)

