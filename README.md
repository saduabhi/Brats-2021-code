# Brats-2021-code


def Hausdorff_distance(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
        
    print(type(y_pred_f))
    # first method
    d2_matrix = pairwise_distances(y_true_f,y_pred_f, metric ='euclidean')
    result = tf.math.reduce_mean(tf.math.minimum(d2_matrix, axis=0)) + tf.math.reduce_mean(tf.math.minimum(d2_matrix, axis=1))
    return result



https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net#Problem-definiton
