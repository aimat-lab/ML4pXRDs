import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    icsd_info = pd.read_csv(
        "/home/henrik/Dokumente/Big_Files/ICSD/ICSD_data_from_API.csv",
        sep=",",
        skiprows=1,
    )

    structure_types = np.array(list(icsd_info["StructureType"]))
    years = list(icsd_info["PublicationYear"])

    years_unique = list(set(years))
    years_unique.sort()

    structure_types_so_far = []

    structure_types_over_year = []
    structures_over_year = []

    additional_structures_over_year = []
    additional_structure_types_over_year = []

    for i, year in enumerate(years_unique):

        additional_structures = years.count(year)
        additional_structures_over_year.append(additional_structures)
        structures_over_year.append(
            (structures_over_year[i - 1] if i != 0 else 0) + additional_structures
        )

        structure_types_of_year = list(
            np.unique(structure_types[np.array(years) == year])
        )

        NO_structure_types_before = len(structure_types_so_far)

        structure_types_so_far.extend(structure_types_of_year)
        structure_types_so_far = list(np.unique(structure_types_so_far))

        NO_structure_types_after = len(structure_types_so_far)
        additional_structure_types_over_year.append(
            NO_structure_types_after - NO_structure_types_before
        )

        structure_types_over_year.append(len(structure_types_so_far))

    plt.plot(years_unique, structures_over_year, label="Number of structures")
    plt.plot(years_unique, structure_types_over_year, label="Number of structure types")
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Number")
    plt.savefig("ICSD_evolution.pdf")
    plt.show()

    # plt.plot(
    #    years_unique,
    #    additional_structures_over_year,
    #    label="Additional structures of that year",
    # )
    plt.plot(
        years_unique,
        additional_structure_types_over_year,
        label="Additional structure types of that year",
    )
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Number")
    plt.savefig("ICSD_evolution_additional.pdf")
    plt.show()
