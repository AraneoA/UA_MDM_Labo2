import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

from json import loads

def get_artifact_filename(study, prefix):
    for k in study.best_trial.system_attrs.keys():
        artifact = loads(study.best_trial.system_attrs[k])
        if artifact['filename'][0:len(prefix)]==prefix:
            return(artifact['artifact_id'])


def plot_confusion_matrix(y_test, y_pred, labels=None, title='Conf Matrix', counts=True):

    if labels is None:
        labels = np.unique(y_test)

    cm = confusion_matrix(y_test, y_pred, labels = labels, normalize='true')
    cm_counts = confusion_matrix(y_test, y_pred, labels = labels)

    hm = go.Heatmap(z=cm*100, y=labels, x=labels)

    annotations = []
    for i, row in enumerate(cm):  #iterate through true labels i
        for j, value in enumerate(row): # iterate through predictions
            if counts:
                ann_text = str(np.round(value*100,1)) + ' - ' + str(round(cm_counts[i,j]))
            else:
                ann_text = str(np.round(value*100,1)) 
                    
            annotations.append(
                {
                    "x": labels[j],
                    "y": labels[i],
                    "font": {"color": "white"},
                    "text": ann_text,
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
        'width':800,
        'height':800,        
        "annotations": annotations
    }
    fig = go.Figure(data=hm, layout=layout, )
    return fig