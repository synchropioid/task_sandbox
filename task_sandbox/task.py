# sys import
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# third party import
import nibabel
from nistats.design_matrix import make_design_matrix, check_design_matrix
from nistats.first_level_model import FirstLevelModel
from nilearn.image import index_img, load_img
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc
from pypreprocess.reporting.glm_reporter import generate_subject_stats_report

# package import
from task_sandbox.config import root_dir, data_dir, func_data_filename

# global definition
jobfile = os.path.join(root_dir, "task.ini")
output_dir = "analysis_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#########################################################################
# Preprocessing
subject_data = do_subjects_preproc(jobfile, dataset_dir=data_dir)[0]
fmri_img = load_img(subject_data.func[0])

#########################################################################
# Seed base analysis GLM
# construct experimental paradigm
stats_start_time = time.ctime()
tr = 2.
n_scans = fmri_img.shape[-1]
epoch_duration = 30.
conditions = ['rest', 'tapping_right', 'rest', 'tapping_left'] * 5
duration = np.array([epoch_duration] * len(conditions))
nb_conditions = len(conditions)
onset = np.linspace(0, (nb_conditions - 1) * epoch_duration, nb_conditions)
paradigm = pd.DataFrame({'onset': onset,
                         'duration': duration,
                         'trial_type': conditions})
hfcut = 2 * 2 * epoch_duration
# construct design matrix
frametimes = np.linspace(0, (n_scans - 1) * tr, n_scans)
drift_model = 'Cosine'
hrf_model = 'spm'
design_matrix = make_design_matrix(frametimes, paradigm,
                                   hrf_model=hrf_model,
                                   drift_model=drift_model,
                                   period_cut=hfcut)
# specify contrasts
_, _, names = check_design_matrix(design_matrix)
contrast_matrix = np.eye(len(names))
contrasts = dict([(names[i], contrast_matrix[i]) for i in range(len(names))])

# more interesting contrasts
tapping_left = contrasts['tapping_left'] - contrasts['tapping_right']
tapping_right = contrasts['tapping_right'] - contrasts['tapping_left']
contrasts = {'tapping_left': tapping_left,
             'tapping_right': tapping_right,
             }
# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FirstLevelModel(t_r=tr, slice_time_ref=0.5, noise_model='ar1',
                           standardize=False)
fmri_glm.fit(run_imgs=fmri_img, design_matrices=design_matrix)
# save computed mask
mask_path = os.path.join(subject_data.output_dir, "mask.nii.gz")
print("Saving mask image {0}".format(mask_path))
nibabel.save(fmri_glm.masker_.mask_img_, mask_path)
# compute bg unto which activation will be projected
anat_img = load_img(subject_data.anat)
print("Computing contrasts ..")
z_maps = {}
for contrast_id, contrast_val in contrasts.items():
    print("\tcontrast id: {0}".format(contrast_id))
    z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')
    # store stat maps to disk
    map_dir = os.path.join(output_dir, 'z_maps')
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    map_path = os.path.join(map_dir, '{0}.nii.gz'.format(contrast_id))
    nibabel.save(z_map, map_path)
    # collect zmaps for contrasts we're interested in
    z_maps[contrast_id] = map_path
    print("\t\tz map: {0}".format(map_path))
# do stats report
stats_report_filename = os.path.join(subject_data.reports_output_dir,
                                     "report_stats.html")
contrasts = dict((contrast_id, contrasts[contrast_id])
                                          for contrast_id in z_maps.keys())
# hack because of the pypreprocess' internal outdated Nistats
paradigm['name'] = paradigm.pop('trial_type')
generate_subject_stats_report(stats_report_filename, contrasts, z_maps,
                              fmri_glm.masker_.mask_img_,
                              design_matrices=[design_matrix],
                              subject_id=subject_data.subject_id,
                              anat=anat_img, display_mode='ortho',
                              threshold=3.0, cluster_th=50, # cluster analysis params
                              start_time=stats_start_time,
                              paradigm=paradigm, TR=tr, nscans=n_scans,
                              hfcut=hfcut, frametimes=frametimes,
                              drift_model=drift_model, hrf_model=hrf_model)
print("\r\nStatistic report written to {0}\r\n".format(stats_report_filename))
