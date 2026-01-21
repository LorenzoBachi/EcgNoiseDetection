import h5py
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

def load_noiseIdxDB_features():
    cwd = Path.cwd()
    parent = cwd.parent
    path = parent.joinpath("EcgNoiseData","noiseIdxDB_features.h5")

    with h5py.File(path, "r") as f:
        dbFeatures  = f["dbFeatures"][:]
        featureNames = f["featureNames"][()].astype(str).tolist()
        featureNames = [f[0] if isinstance(f, (list, tuple)) else f for f in featureNames]
        labels  = f["labels"][:]
        labelNames = f["labelNames"][()].astype(str).tolist()

    X = dbFeatures.T    # data matrix
    y = labels[0,:]     # main outcome
    a = labels[1,:]     # annotation / additional outcomes (a = 3 for ventricula arrhythmias, a = 4 for supraventricular tachycardia)
    nr = X.shape[0]     # number of samples
    nf = X.shape[1]     # number of features
    print("Noise index database: " + str(nr) + " samples, " + str(nf) + " features ")
    print("Shape of main outcome y: " + str(y.shape))
    # unique subject IDs for GroupKFold
    subject_ids = (labels[2,:]*10000 ) + labels[3,:]
    return X, y, a, featureNames, labelNames, subject_ids

def compute_noiseIdxDB_sample_weights(a,y):
    w = np.ones(y.shape)
    nn = np.sum(a==0)   # number of normal ECG records
    no = np.sum(a==1)   # number of noisy ECG records
    nv = np.sum(a==3)   # number of ECG records with ventricular arrhythmias
    nt = np.sum(a==4)   # number of ECG records with supraventricular tachycardias
    # compute weights inversely proportional to class frequencies. Normal ECG records have weight 1
    wo = nn/no  # weight for noisy records
    wv = nn/nv  # weight for ventricular arrhythmias
    wt = nn/nt  # weight for supraventricular tachycardias
    print("Normal ECG records: " + str(nn) + ", weight = 1")
    print("Noise records: " + str(no) + ", weight = " + str(wo))
    print("Records of ECG with ventricular arrhythmias: " + str(nv) + ", weight = " + str(wv))
    print("Records of ECG with supraventricular tachycardias: " + str(nt) + ", weight = " + str(wt))
    w[a==1] = wo
    w[a==3] = wv
    w[a==4] = wt
    return w

def get_performance_metrics(y, y_pred, a):
    MCC = matthews_corrcoef(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    mask = ( a== 3)
    y_true_v = y[mask]
    y_pred_v = y_pred[mask]
    cm_sub = confusion_matrix(y_true_v, y_pred_v)
    TN_v, FP_v = cm_sub[0,0], cm_sub[0,1]
    spec_v = TN_v / (TN_v + FP_v)
    return MCC, sens, spec, spec_v
