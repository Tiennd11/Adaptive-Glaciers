from dataclasses import dataclass, asdict
from tabulate import tabulate

@dataclass
class Params:
    # Geometry
    dimension: int = 3
    plane_stress: bool = False  # Only relevant for 2D problems

    # Domain dimensions in meters
    Lx: float = 500
    Ly: float = 750
    Lz: float = 125

    # Physical constants
    g: float = 9.81
    rho_sea: float = 1020 
    E0_ice: float = 9500e6
    nu_ice: float = 0.35
    rho_i_ice: float = 917 
    sigma_c_ice: float =  0.1185e6
    ci: float = 1

    # Fracture parameters
    l: float = 0.625  # Length scale for fracture propagation, in meters



    # Units metadata
    _units = {
        "dimension": "-",
        "plane_stress": "Bool",
        "Lx": "m",
        "Ly": "m",
        "Lz": "m",
        "g": "m/s²",
        "rho_sea": "N/m³",
        "E0_ice": "Pa",
        "nu_ice": "-",
        "rho_i_ice": "N/m³",
        "sigma_c_ice": "Pa",
        "ci": "-",
        "l": "m",
    }

    def print_table(self, tablefmt: str = "github"):
        """
        Print all parameters in a table format using tabulate.
        You can specify the table format (e.g., 'grid', 'plain', 'github').
        """
        data = []
        for k, v in asdict(self).items():
            unit = self._units.get(k, "")
            data.append((k, v, unit))
        print(tabulate(data, headers=["Parameter", "Value", "Unit"], tablefmt=tablefmt))