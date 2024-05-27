import os
from mp_api.client import MPRester
from monty.serialization import dumpfn


save_path = "/Users/hm-t03-mac2/Documents/py_proj/DosNet/data/"
API_KEY='WSixK0Or5Ro47ND37DnNcd49xanMqUsV' # API key from materials project login dashboard online

mpr = MPRester(API_KEY)
data = mpr.summary.search(fields=["material_id", 
                                      "formation_energy_per_atom"])
dumpfn(data, os.path.join(save_path, f"mp_doc.json.gz"))

#print(len(data), data[0])

'''
155361 MPDataDoc<SummaryDoc>
material_id=MPID(mp-1056831),
formation_energy_per_atom=1.3098259412500006

Fields not requested:
['builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name', 'deprecated', 'deprecation_reasons', 'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 'has_props', 'theoretical', 'database_IDs']
'''